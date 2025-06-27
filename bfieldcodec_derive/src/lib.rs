//! This crate provides a derive macro for the `BFieldCodec` trait.
//!
//! The macro emits statements starting with:
//!   `use crate::twenty_first::`
//!
//! Crates that use this macro must add a use statement `twenty_first`
//! in their lib.rs:
//!   `use twenty_first;`
//!
//! or if using twenty-first via re-export in `dep_crate`:
//!   `use dep_crate::twenty_first;`
//!
//! Failure to do so will result in compile errors.
//
// See the corresponding attribute in twenty_first/lib.rs
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

extern crate proc_macro;

use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::quote;
use syn::Attribute;
use syn::DeriveInput;
use syn::Field;
use syn::Fields;
use syn::Ident;
use syn::Type;
use syn::Variant;
use syn::parse_macro_input;
use syn::punctuated::Punctuated;
use syn::token::Comma;

/// Derives `BFieldCodec` for structs and enums.
///
/// Fields that should not be serialized can be ignored by annotating them with
/// `#[bfield_codec(ignore)]`.
/// Ignored fields must implement [`Default`].
///
/// For enums, the discriminant used for serialization can be accessed through method
/// `bfield_codec_discriminant`.
///
/// ### Example
///
/// ```ignore
/// #[derive(BFieldCodec)]
/// struct Foo {
///    bar: u64,
///    #[bfield_codec(ignore)]
///    ignored: usize,
/// }
/// let foo = Foo { bar: 42, ignored: 7 };
/// let encoded = foo.encode();
/// let decoded = Foo::decode(&encoded).unwrap();
/// assert_eq!(foo.bar, decoded.bar);
/// ```
///
/// Accessing the discriminant of an enum's variant:
///
/// ```ignore
/// #[derive(BFieldCodec)]
/// enum Bar {
///     Baz,
///     Qux(u64),
/// }
/// let _discriminant = Bar::Baz.bfield_codec_discriminant();
/// ```
///
/// ### Known limitations
///
/// - Enums with variants that have named fields are currently not supported. Example:
///      ```ignore
///     #[derive(BFieldCodec)]  // Currently not supported.
///     enum Foo {
///        Bar { baz: u64 },
///     }
///     ```
///
/// - Enums with no variants are currently not supported. Consider using a unit struct instead.
///   Example:
///     ```ignore
///     #[derive(BFieldCodec)]  // Currently not supported.
///     enum Foo {}             // Consider `struct Foo;` instead.
///     ```
///
/// - Ignoring fields in tuple structs is currently not supported. Consider
///   using a struct with named fields instead. Example:
///   ```ignore
///   #[derive(BFieldCodec)]
///   struct Foo(#[bfield_codec(ignore)] u32);
///   //         ~~~~~~~~~~~~~~~~~~~~~~~
///   //         Currently not supported in tuple structs
///   ```
#[proc_macro_derive(BFieldCodec, attributes(bfield_codec))]
pub fn bfieldcodec_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    BFieldCodecDeriveBuilder::new(ast).build().into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BFieldCodecDeriveType {
    UnitStruct,
    StructWithNamedFields,
    StructWithUnnamedFields,
    Enum,
}

struct BFieldCodecDeriveBuilder {
    name: Ident,
    derive_type: BFieldCodecDeriveType,
    generics: syn::Generics,
    attributes: Vec<Attribute>,

    named_included_fields: Vec<Field>,
    named_ignored_fields: Vec<Field>,

    unnamed_fields: Vec<Field>,

    variants: Option<Punctuated<Variant, syn::token::Comma>>,

    encode_statements: Vec<TokenStream>,
    decode_function_body: TokenStream,
    static_length_body: TokenStream,
    error_builder: BFieldCodecErrorEnumBuilder,
}

struct BFieldCodecErrorEnumBuilder {
    name: Ident,
    errors: HashMap<&'static str, BFieldCodecErrorEnumVariant>,
}

struct BFieldCodecErrorEnumVariant {
    variant_name: Ident,
    variant_type: TokenStream,
    display_match_arm: TokenStream,
}

impl BFieldCodecDeriveBuilder {
    fn new(ast: DeriveInput) -> Self {
        let derive_type = Self::extract_derive_type(&ast);

        let named_fields = Self::extract_named_fields(&ast);
        let (ignored_fields, included_fields) = named_fields
            .iter()
            .cloned()
            .partition::<Vec<_>, _>(Self::field_is_ignored);

        let unnamed_fields = Self::extract_unnamed_fields(&ast);
        let variants = Self::extract_variants(&ast);

        let name = ast.ident;
        let error_builder = BFieldCodecErrorEnumBuilder::new(name.clone());

        Self {
            name,
            derive_type,
            generics: ast.generics,
            attributes: ast.attrs,

            named_included_fields: included_fields,
            named_ignored_fields: ignored_fields,
            unnamed_fields,
            variants,

            encode_statements: vec![],
            decode_function_body: quote! {},
            static_length_body: quote! {},
            error_builder,
        }
    }

    fn extract_derive_type(ast: &DeriveInput) -> BFieldCodecDeriveType {
        match &ast.data {
            syn::Data::Struct(syn::DataStruct {
                fields: Fields::Unit,
                ..
            }) => BFieldCodecDeriveType::UnitStruct,
            syn::Data::Struct(syn::DataStruct {
                fields: Fields::Named(_),
                ..
            }) => BFieldCodecDeriveType::StructWithNamedFields,
            syn::Data::Struct(syn::DataStruct {
                fields: Fields::Unnamed(_),
                ..
            }) => BFieldCodecDeriveType::StructWithUnnamedFields,
            syn::Data::Enum(_) => BFieldCodecDeriveType::Enum,
            _ => panic!("expected a struct or an enum"),
        }
    }

    fn extract_named_fields(ast: &DeriveInput) -> Vec<Field> {
        match &ast.data {
            syn::Data::Struct(syn::DataStruct {
                fields: Fields::Named(fields),
                ..
            }) => fields.named.iter().rev().cloned().collect::<Vec<_>>(),
            _ => vec![],
        }
    }

    fn extract_unnamed_fields(ast: &DeriveInput) -> Vec<Field> {
        match &ast.data {
            syn::Data::Struct(syn::DataStruct {
                fields: Fields::Unnamed(fields),
                ..
            }) => fields.unnamed.iter().cloned().collect::<Vec<_>>(),
            _ => vec![],
        }
    }

    fn extract_variants(ast: &DeriveInput) -> Option<Punctuated<Variant, Comma>> {
        match &ast.data {
            syn::Data::Enum(data_enum) => Some(data_enum.variants.clone()),
            _ => None,
        }
    }

    fn field_is_ignored(field: &Field) -> bool {
        let field_name = field.ident.as_ref().unwrap();
        let mut relevant_attributes = field
            .attrs
            .iter()
            .filter(|attr| attr.path().is_ident("bfield_codec"));

        let Some(attribute) = relevant_attributes.next() else {
            return false;
        };
        if relevant_attributes.next().is_some() {
            panic!("field `{field_name}` must have at most 1 `bfield_codec` attribute");
        }

        let parse_ignore = attribute.parse_nested_meta(|meta| match meta.path.get_ident() {
            Some(ident) if ident == "ignore" => Ok(()),
            Some(ident) => panic!("unknown identifier `{ident}` for field `{field_name}`"),
            _ => unreachable!(),
        });
        parse_ignore.is_ok()
    }

    fn build(mut self) -> TokenStream {
        self.error_builder.build(self.derive_type);
        self.add_trait_bounds_to_generics();
        self.build_methods();
        self.into_tokens()
    }

    fn add_trait_bounds_to_generics(&mut self) {
        let ignored_generics = self.extract_ignored_generics_list();
        let ignored_generics = self.recursively_collect_all_ignored_generics(ignored_generics);

        for param in &mut self.generics.params {
            let syn::GenericParam::Type(type_param) = param else {
                continue;
            };
            if ignored_generics.contains(&type_param.ident) {
                continue;
            }
            type_param.bounds.push(syn::parse_quote!(BFieldCodec));
        }
    }

    fn extract_ignored_generics_list(&self) -> Vec<syn::Ident> {
        self.attributes
            .iter()
            .flat_map(Self::extract_ignored_generics)
            .collect()
    }

    fn extract_ignored_generics(attr: &Attribute) -> Vec<Ident> {
        if !attr.path().is_ident("bfield_codec") {
            return vec![];
        }

        let mut ignored_generics = vec![];
        attr.parse_nested_meta(|meta| match meta.path.get_ident() {
            Some(ident) if ident == "ignore" => {
                ignored_generics.push(ident.to_owned());
                Ok(())
            }
            Some(ident) => Err(meta.error(format!("Unknown identifier \"{ident}\"."))),
            _ => Err(meta.error("Expected an identifier.")),
        })
        .unwrap();
        ignored_generics
    }

    /// For all ignored fields, add all type identifiers (including, recursively, the type
    /// identifiers of generic type arguments) to the list of ignored type identifiers.
    fn recursively_collect_all_ignored_generics(
        &self,
        mut ignored_generics: Vec<Ident>,
    ) -> Vec<Ident> {
        let mut ignored_types = self
            .named_ignored_fields
            .iter()
            .map(|ignored_field| ignored_field.ty.clone())
            .collect::<Vec<_>>();
        while !ignored_types.is_empty() {
            let ignored_type = ignored_types[0].clone();
            ignored_types = ignored_types[1..].to_vec();
            let Type::Path(type_path) = ignored_type else {
                continue;
            };
            for segment in type_path.path.segments.into_iter() {
                ignored_generics.push(segment.ident);
                let syn::PathArguments::AngleBracketed(generic_arguments) = segment.arguments
                else {
                    continue;
                };
                for generic_argument in generic_arguments.args.into_iter() {
                    let syn::GenericArgument::Type(t) = generic_argument else {
                        continue;
                    };
                    ignored_types.push(t.clone());
                }
            }
        }
        ignored_generics
    }

    fn build_methods(&mut self) {
        match self.derive_type {
            BFieldCodecDeriveType::UnitStruct => self.build_methods_for_unit_struct(),
            BFieldCodecDeriveType::StructWithNamedFields => {
                self.build_methods_for_struct_with_named_fields()
            }
            BFieldCodecDeriveType::StructWithUnnamedFields => {
                self.build_methods_for_struct_with_unnamed_fields()
            }
            BFieldCodecDeriveType::Enum => self.build_methods_for_enum(),
        }
    }

    fn build_methods_for_unit_struct(&mut self) {
        self.build_decode_function_body_for_unit_struct();
        self.static_length_body = quote! {::core::option::Option::Some(0)};
    }

    fn build_methods_for_struct_with_named_fields(&mut self) {
        self.build_encode_statements_for_struct_with_named_fields();
        self.build_decode_function_body_for_struct_with_named_fields();
        let included_fields = self.named_included_fields.clone();
        self.build_static_length_body_for_struct(&included_fields);
    }

    fn build_methods_for_struct_with_unnamed_fields(&mut self) {
        self.build_encode_statements_for_struct_with_unnamed_fields();
        self.build_decode_function_body_for_struct_with_unnamed_fields();
        let included_fields = self.unnamed_fields.clone();
        self.build_static_length_body_for_struct(&included_fields);
    }

    fn build_methods_for_enum(&mut self) {
        self.build_encode_statements_for_enum();
        self.build_decode_function_body_for_enum();
        self.build_static_length_body_for_enum();
    }

    fn build_encode_statements_for_struct_with_named_fields(&mut self) {
        let included_field_names = self
            .named_included_fields
            .iter()
            .map(|field| field.ident.as_ref().unwrap().to_owned());
        let included_field_types = self
            .named_included_fields
            .iter()
            .map(|field| field.ty.clone());
        self.encode_statements = included_field_names
            .clone()
            .zip(included_field_types.clone())
            .map(|(field_name, field_type)| {
                quote! {
                    let #field_name:
                        ::std::vec::Vec<crate::twenty_first::prelude::BFieldElement>
                            = self.#field_name.encode();
                    if <#field_type as crate::twenty_first::prelude::BFieldCodec>
                        ::static_length().is_none() {
                        elements.push(
                            crate::twenty_first::prelude::BFieldElement::new(
                                #field_name.len() as u64
                            )
                        );
                    }
                    elements.extend(#field_name);
                }
            })
            .collect();
    }

    fn build_encode_statements_for_struct_with_unnamed_fields(&mut self) {
        let field_types = self.unnamed_fields.iter().map(|field| field.ty.clone());
        let indices: Vec<_> = (0..self.unnamed_fields.len())
            .map(syn::Index::from)
            .collect();
        let field_names: Vec<_> = indices
            .iter()
            .map(|i| quote::format_ident!("field_value_{}", i.index))
            .collect();
        self.encode_statements = indices
            .iter()
            .zip(field_types.clone())
            .zip(field_names.clone())
            .rev()
            .map(|((idx, field_type), field_name)| {
                quote! {
                    let #field_name:
                        ::std::vec::Vec<crate::twenty_first::prelude::BFieldElement>
                            = self.#idx.encode();
                    if <#field_type as crate::twenty_first::prelude::BFieldCodec>
                        ::static_length().is_none() {
                        elements.push(
                            crate::twenty_first::prelude::BFieldElement::new(
                                #field_name.len() as u64
                            )
                        );
                    }
                    elements.extend(#field_name);
                }
            })
            .collect();
    }

    fn build_encode_statements_for_enum(&mut self) {
        let encode_clauses = self
            .enum_discriminants_and_variants()
            .into_iter()
            .map(|(d, v)| self.generate_encode_clause_for_variant(d, v));
        let encode_match_statement = quote! {
            match self {
                #( #encode_clauses , )*
            }
        };
        self.encode_statements = vec![encode_match_statement];
    }

    fn generate_encode_clause_for_variant(
        &self,
        discriminant: usize,
        variant: &Variant,
    ) -> TokenStream {
        let variant_name = &variant.ident;
        let associated_data = &variant.fields;

        if associated_data.is_empty() {
            return quote! {
                Self::#variant_name => {
                    elements.push(crate::twenty_first::prelude::BFieldElement::new(
                        #discriminant as u64)
                    );
                }
            };
        }

        let reversed_enumerated_associated_data = associated_data.iter().enumerate().rev();
        let field_encoders = reversed_enumerated_associated_data.map(|(field_index, ad)| {
            let field_name = self.enum_variant_field_name(discriminant, field_index);
            let field_type = ad.ty.clone();
            let field_encoding =
                quote::format_ident!("variant_{}_field_{}_encoding", discriminant, field_index);
            quote! {
                let #field_encoding:
                    ::std::vec::Vec<crate::twenty_first::prelude::BFieldElement> =
                        #field_name.encode();
                if <#field_type as crate::twenty_first::prelude::BFieldCodec>
                    ::static_length().is_none() {
                    elements.push(
                        crate::twenty_first::prelude::BFieldElement::new(
                            #field_encoding.len() as u64
                        )
                    );
                }
                elements.extend(#field_encoding);
            }
        });

        let field_names = associated_data
            .iter()
            .enumerate()
            .map(|(field_index, _field)| self.enum_variant_field_name(discriminant, field_index));

        quote! {
            Self::#variant_name ( #( #field_names , )* ) => {
                elements.push(
                    crate::twenty_first::prelude::BFieldElement::new(
                        #discriminant as u64
                    )
                );
                #( #field_encoders )*
            }
        }
    }

    fn build_decode_function_body_for_unit_struct(&mut self) {
        let sequence_too_long_error = self.error_builder.sequence_too_long();

        self.decode_function_body = quote! {
            if !sequence.is_empty() {
                return ::core::result::Result::Err(#sequence_too_long_error(sequence.len()));
            }
            ::core::result::Result::Ok(::std::boxed::Box::new(Self))
        };
    }

    fn build_decode_function_body_for_struct_with_named_fields(&mut self) {
        let sequence_too_long_error = self.error_builder.sequence_too_long();

        let decode_statements = self
            .named_included_fields
            .iter()
            .map(|field| {
                let field_name = field.ident.as_ref().unwrap();
                self.generate_decode_statement_for_field(field_name, &field.ty)
            })
            .collect::<Vec<_>>();

        let included_field_names = self.named_included_fields.iter().map(|field| {
            let field_name = field.ident.as_ref().unwrap().to_owned();
            quote! { #field_name }
        });
        let ignored_field_names = self.named_ignored_fields.iter().map(|field| {
            let field_name = field.ident.as_ref().unwrap().to_owned();
            quote! { #field_name }
        });

        self.decode_function_body = quote! {
            #(#decode_statements)*
            if !sequence.is_empty() {
                return ::core::result::Result::Err(#sequence_too_long_error(sequence.len()));
            }
            ::core::result::Result::Ok(::std::boxed::Box::new(Self {
                #(#included_field_names,)*
                #(#ignored_field_names: ::core::default::Default::default(),)*
            }))
        };
    }

    fn build_decode_function_body_for_struct_with_unnamed_fields(&mut self) {
        let sequence_too_long_error = self.error_builder.sequence_too_long();

        let field_names = (0..self.unnamed_fields.len())
            .map(|i| quote::format_ident!("field_value_{}", i))
            .collect::<Vec<_>>();
        let decode_statements = field_names
            .iter()
            .zip(self.unnamed_fields.iter())
            .rev()
            .map(|(field_name, field)| {
                self.generate_decode_statement_for_field(field_name, &field.ty)
            })
            .collect::<Vec<_>>();

        self.decode_function_body = quote! {
            #(#decode_statements)*
            if !sequence.is_empty() {
                return ::core::result::Result::Err(#sequence_too_long_error(sequence.len()));
            }
            ::core::result::Result::Ok(::std::boxed::Box::new(Self ( #(#field_names,)* )))
        };
    }

    fn generate_decode_statement_for_field(
        &self,
        field_name: &Ident,
        field_type: &Type,
    ) -> TokenStream {
        let sequence_empty_for_field_error = self.error_builder.sequence_empty_for_field();
        let sequence_too_short_for_field_error = self.error_builder.sequence_too_short_for_field();
        let field_name_as_string_literal = field_name.to_string();
        quote! {
            let (#field_name, sequence) = {
                let maybe_fields_static_length =
                    <#field_type as crate::twenty_first::prelude::BFieldCodec>
                        ::static_length();
                let field_has_dynamic_length = maybe_fields_static_length.is_none();
                if sequence.is_empty() && field_has_dynamic_length {
                    return ::core::result::Result::Err(
                        #sequence_empty_for_field_error(#field_name_as_string_literal.to_string())
                    );
                }
                let (len, sequence) = match maybe_fields_static_length {
                    ::core::option::Option::Some(len) => (len, sequence),
                    ::core::option::Option::None => (sequence[0].value() as usize, &sequence[1..]),
                };
                if sequence.len() < len {
                    return ::core::result::Result::Err(#sequence_too_short_for_field_error(
                        #field_name_as_string_literal.to_string(),
                    ));
                }
                let decoded =
                    *<#field_type as crate::twenty_first::prelude::BFieldCodec>
                        ::decode(&sequence[..len]).map_err(|err|
                            -> ::std::boxed::Box<
                                    dyn ::std::error::Error
                                    + ::core::marker::Send
                                    + ::core::marker::Sync
                            > {
                                err.into()
                            }
                        )?;
                (decoded, &sequence[len..])
            };
        }
    }

    fn build_decode_function_body_for_enum(&mut self) {
        let sequence_empty_error = self.error_builder.sequence_empty();
        let invalid_variant_error = self.error_builder.invalid_discriminant();

        let mut match_arms = vec![];
        for (discriminant, variant) in self.enum_discriminants_and_variants() {
            let decode_clause = self.generate_decode_clause_for_variant(discriminant, variant);
            let match_arm = quote! { #discriminant => { #decode_clause } };
            match_arms.push(match_arm);
        }

        self.decode_function_body = quote! {
            if sequence.is_empty() {
                return ::core::result::Result::Err(#sequence_empty_error);
            }
            let (discriminant, sequence) = (sequence[0].value() as usize, &sequence[1..]);
            match discriminant {
                #(#match_arms ,)*
                other_index => ::core::result::Result::Err(#invalid_variant_error(other_index)),
            }
        };
    }

    fn generate_decode_clause_for_variant(
        &self,
        discriminant: usize,
        variant: &Variant,
    ) -> TokenStream {
        let sequence_too_long_error = self.error_builder.sequence_too_long();
        let sequence_empty_error = self.error_builder.sequence_empty_for_variant();
        let sequence_too_short_error = self.error_builder.sequence_too_short_for_variant();

        let variant_name = &variant.ident;
        let associated_data = &variant.fields;
        if associated_data.is_empty() {
            return quote! {
                if !sequence.is_empty() {
                    return ::core::result::Result::Err(#sequence_too_long_error(sequence.len()));
                }
                ::core::result::Result::Ok(::std::boxed::Box::new(Self::#variant_name))
            };
        }

        let field_decoders = associated_data
            .iter()
            .enumerate()
            .rev()
            .map(|(field_index, field)| {
                let field_type = field.ty.clone();
                let field_name = self.enum_variant_field_name(discriminant, field_index);
                let field_value =
                    quote::format_ident!("variant_{}_field_{}_value", discriminant, field_index);
                quote! {
                    let (#field_value, sequence) = {
                        let maybe_fields_static_length =
                            <#field_type as crate::twenty_first::prelude::BFieldCodec>
                                ::static_length();
                        let field_has_dynamic_length = maybe_fields_static_length.is_none();
                        if sequence.is_empty() && field_has_dynamic_length {
                            return ::core::result::Result::Err(
                                #sequence_empty_error(#discriminant, #field_index)
                            );
                        }
                        let (len, sequence) = match maybe_fields_static_length {
                            ::core::option::Option::Some(len) => (len, sequence),
                            ::core::option::Option::None => {
                                (sequence[0].value() as usize, &sequence[1..])
                            },
                        };
                        if sequence.len() < len {
                            return ::core::result::Result::Err(
                                #sequence_too_short_error(#discriminant, #field_index)
                            );
                        }
                        let decoded =
                            *<#field_type as crate::twenty_first::prelude::BFieldCodec>
                                ::decode(
                                    &sequence[..len]
                                ).map_err(|err|
                                    -> ::std::boxed::Box<
                                            dyn ::std::error::Error
                                            + ::core::marker::Send
                                            + ::core::marker::Sync
                                    > {
                                        err.into()
                                    }
                                )?;
                        (decoded, &sequence[len..])
                    };
                    let #field_name = #field_value;
                }
            })
            .fold(quote! {}, |l, r| quote! {#l #r});
        let field_names = associated_data
            .iter()
            .enumerate()
            .map(|(field_index, _field)| self.enum_variant_field_name(discriminant, field_index));
        quote! {
            #field_decoders
            if !sequence.is_empty() {
                return ::core::result::Result::Err(#sequence_too_long_error(sequence.len()));
            }
            ::core::result::Result::Ok(
                ::std::boxed::Box::new(Self::#variant_name ( #( #field_names , )* ))
            )
        }
    }

    fn enum_variant_field_name(&self, discriminant: usize, field_index: usize) -> syn::Ident {
        quote::format_ident!("variant_{}_field_{}", discriminant, field_index)
    }

    fn build_static_length_body_for_struct(&mut self, fields: &[Field]) {
        let field_types = fields
            .iter()
            .map(|field| field.ty.clone())
            .collect::<Vec<_>>();
        let num_fields = field_types.len();
        self.static_length_body = quote! {
            let field_lengths : [::core::option::Option<usize>; #num_fields] = [
                #(
                    <#field_types as
                    crate::twenty_first::prelude::BFieldCodec>::static_length(),
                )*
            ];
            if field_lengths.iter().all(|fl| fl.is_some() ) {
                ::core::option::Option::Some(field_lengths.iter().map(|fl| fl.unwrap()).sum())
            }
            else {
                ::core::option::Option::None
            }
        };
    }

    fn build_static_length_body_for_enum(&mut self) {
        let variants = self.variants.as_ref().unwrap();
        let no_variants_have_associated_data = variants.iter().all(|v| v.fields.is_empty());
        if no_variants_have_associated_data {
            self.static_length_body = quote! {::core::option::Option::Some(1)};
            return;
        }

        let num_variants = variants.len();
        if num_variants == 0 {
            self.static_length_body = quote! {::core::option::Option::Some(0)};
            return;
        }

        // some variants have associated data
        // if all variants encode to the same length, the length is statically known anyway
        let variant_lengths = variants
            .iter()
            .map(|variant| {
                let fields = variant.fields.clone();
                let field_lengths = fields.iter().map(|f| {
                    quote! {
                        <#f as crate::twenty_first::prelude::BFieldCodec>
                            ::static_length()
                    }
                });
                let num_fields = fields.len();
                quote! {{
                    let field_lengths: [::core::option::Option<usize>; #num_fields] =
                        [ #( #field_lengths , )* ];
                    if field_lengths.iter().all(|fl| fl.is_some()) {
                        Some(field_lengths.iter().map(|fl|fl.unwrap()).sum())
                    } else {
                        None
                    }
                }}
            })
            .collect::<Vec<_>>();

        self.static_length_body = quote! {
                let variant_lengths : [::core::option::Option<usize>; #num_variants] =
                    [ #( #variant_lengths , )* ];
                if variant_lengths.iter().all(|field_len| field_len.is_some()) &&
                    variant_lengths.iter().all(|x| x.unwrap() == variant_lengths[0].unwrap()) {
                    // account for discriminant
                    Some(variant_lengths[0].unwrap() + 1)
                }
                else {
                    None
                }

        };
    }

    fn enum_discriminants_and_variants(&self) -> Vec<(usize, &Variant)> {
        self.variants.as_ref().unwrap().iter().enumerate().collect()
    }

    fn maybe_impl_enum_discriminants(&self) -> TokenStream {
        if self.derive_type != BFieldCodecDeriveType::Enum {
            return quote! {};
        }

        let mut variant_match_arms = vec![];
        for (discriminant, variant) in self.enum_discriminants_and_variants() {
            let ident = &variant.ident;
            let mut match_statement = quote! { Self::#ident };
            if !variant.fields.is_empty() {
                match_statement.extend(quote! { ( .. ) });
            }
            let match_arm = quote! { #match_statement => #discriminant };
            variant_match_arms.push(match_arm);
        }

        let name = self.name.clone();
        let (impl_generics, ty_generics, where_clause) = self.generics.split_for_impl();
        quote! {
            impl #impl_generics #name #ty_generics #where_clause {
                pub fn bfield_codec_discriminant(&self) -> usize {
                    match self {
                        #( #variant_match_arms , )*
                    }
                }
            }
        }
    }

    fn into_tokens(self) -> TokenStream {
        let maybe_impl_enum_discriminants = self.maybe_impl_enum_discriminants();
        let name = self.name;
        let error_enum_name = self.error_builder.error_enum_name();
        let errors = self.error_builder.into_tokens();
        let decode_function_body = self.decode_function_body;
        let encode_statements = self.encode_statements;
        let static_length_body = self.static_length_body;
        let (impl_generics, ty_generics, where_clause) = self.generics.split_for_impl();

        quote! {
            #maybe_impl_enum_discriminants
            #errors
            impl #impl_generics crate::twenty_first::prelude::BFieldCodec
            for #name #ty_generics #where_clause {
                type Error = #error_enum_name;

                fn decode(
                    sequence: &[crate::twenty_first::prelude::BFieldElement],
                ) -> ::core::result::Result<::std::boxed::Box<Self>, Self::Error> {
                    #decode_function_body
                }

                fn encode(&self) -> ::std::vec::Vec<
                    crate::twenty_first::prelude::BFieldElement
                > {
                    let mut elements = ::std::vec::Vec::new();
                    #(#encode_statements)*
                    elements
                }

                fn static_length() -> ::core::option::Option<usize> {
                    #static_length_body
                }
            }
        }
    }
}

impl BFieldCodecErrorEnumBuilder {
    fn new(name: syn::Ident) -> Self {
        Self {
            name,
            errors: HashMap::new(),
        }
    }

    fn build(&mut self, derive_type: BFieldCodecDeriveType) {
        match derive_type {
            BFieldCodecDeriveType::UnitStruct => self.set_up_unit_struct_errors(),
            BFieldCodecDeriveType::StructWithNamedFields
            | BFieldCodecDeriveType::StructWithUnnamedFields => self.set_up_struct_errors(),
            BFieldCodecDeriveType::Enum => self.set_up_enum_errors(),
        }
    }

    fn set_up_unit_struct_errors(&mut self) {
        self.register_error_sequence_too_long();
        self.register_error_inner_decoding_failure();
    }

    fn set_up_struct_errors(&mut self) {
        self.register_error_sequence_empty();
        self.register_error_sequence_empty_for_field();
        self.register_error_sequence_too_short_for_field();
        self.register_error_sequence_too_long();
        self.register_error_inner_decoding_failure();
    }

    fn set_up_enum_errors(&mut self) {
        self.register_error_sequence_empty();
        self.register_error_sequence_empty_for_variant();
        self.register_error_sequence_too_short_for_variant();
        self.register_error_sequence_too_long();
        self.register_error_invalid_discriminant();
        self.register_error_inner_decoding_failure();
    }

    fn register_error(
        &mut self,
        error_id: &'static str,
        variant_name: Ident,
        variant_type: TokenStream,
        display_match_arm: TokenStream,
    ) {
        self.errors.insert(
            error_id,
            BFieldCodecErrorEnumVariant {
                variant_name,
                variant_type,
                display_match_arm,
            },
        );
    }

    fn global_identifier(&self, variant_name: &Ident) -> TokenStream {
        let error_enum_name = self.error_enum_name();
        quote! { #error_enum_name::#variant_name }
    }

    fn error_enum_name(&self) -> syn::Ident {
        quote::format_ident!("{}BFieldDecodingError", self.name)
    }

    fn register_error_sequence_too_long(&mut self) {
        let name = self.name.to_string();

        let variant_name = quote::format_ident!("SequenceTooLong");
        let variant_type = quote! { #variant_name(usize) };
        let display_match_arm = quote! {
            Self::#variant_name(num_remaining_elements) => ::core::write!(
                f,
                "cannot decode {}: sequence too long ({num_remaining_elements} elements remaining)",
                #name
            )
        };

        self.register_error(
            "seq_too_long",
            variant_name,
            variant_type,
            display_match_arm,
        );
    }

    fn register_error_sequence_empty(&mut self) {
        let name = self.name.to_string();

        let variant_name = quote::format_ident!("SequenceEmpty");
        let variant_type = quote! { #variant_name };
        let display_match_arm = quote! {
            Self::#variant_name => ::core::write!( f, "cannot decode {}: sequence is empty", #name )
        };

        self.register_error("seq_empty", variant_name, variant_type, display_match_arm);
    }

    fn register_error_sequence_empty_for_field(&mut self) {
        let name = self.name.to_string();

        let variant_name = quote::format_ident!("SequenceEmptyForField");
        let variant_type = quote! { #variant_name(String) };
        let display_match_arm = quote! {
            Self::#variant_name(field_name) => ::core::write!(
                f,
                "cannot decode {}, field {field_name}: sequence is empty",
                #name,
            )
        };

        self.register_error(
            "seq_empty_for_field",
            variant_name,
            variant_type,
            display_match_arm,
        );
    }

    fn register_error_sequence_too_short_for_field(&mut self) {
        let name = self.name.to_string();

        let variant_name = quote::format_ident!("SequenceTooShortForField");
        let variant_type = quote! { #variant_name(String) };
        let display_match_arm = quote! {
            Self::#variant_name(field_name) => ::core::write!(
                f,
                "cannot decode {}, field {field_name}: sequence too short",
                #name,
            )
        };

        self.register_error(
            "seq_too_short_for_field",
            variant_name,
            variant_type,
            display_match_arm,
        );
    }

    fn register_error_sequence_empty_for_variant(&mut self) {
        let name = self.name.to_string();

        let variant_name = quote::format_ident!("SequenceEmptyForVariant");
        let variant_type = quote! { #variant_name(usize, usize) };
        let display_match_arm = quote! {
            Self::#variant_name(variant_id, field_id) => ::core::write!(
                f,
                "cannot decode {}, variant {variant_id}, field {field_id}: sequence is empty",
                #name,
            )
        };

        self.register_error(
            "seq_empty_for_variant",
            variant_name,
            variant_type,
            display_match_arm,
        );
    }

    fn register_error_sequence_too_short_for_variant(&mut self) {
        let name = self.name.to_string();

        let variant_name = quote::format_ident!("SequenceTooShortForVariant");
        let variant_type = quote! { #variant_name(usize, usize) };
        let display_match_arm = quote! {
            Self::#variant_name(variant_id, field_id) => ::core::write!(
                f,
                "cannot decode {}, variant {variant_id}, field {field_id}: sequence too short",
                #name,
            )
        };

        self.register_error(
            "seq_too_short_for_variant",
            variant_name,
            variant_type,
            display_match_arm,
        );
    }

    fn register_error_invalid_discriminant(&mut self) {
        let name = self.name.to_string();

        let variant_name = quote::format_ident!("InvalidVariantIndex");
        let variant_type = quote! { #variant_name(usize) };
        let display_match_arm = quote! {
            Self::#variant_name(discriminant) => ::core::write!(
                f,
                "cannot decode {}: invalid variant index {discriminant}",
                #name
            )
        };

        self.register_error(
            "invalid_discriminant",
            variant_name,
            variant_type,
            display_match_arm,
        );
    }

    fn register_error_inner_decoding_failure(&mut self) {
        let name = self.name.to_string();

        let variant_name = quote::format_ident!("InnerDecodingFailure");
        let variant_type = quote! {
            #variant_name(::std::boxed::Box<
                    dyn ::std::error::Error + ::core::marker::Send + ::core::marker::Sync
                >
            )
        };
        let display_match_arm = quote! {
            Self::#variant_name(inner_error) => ::core::write!(
                f,
                "cannot decode {}: inner decoding failure: {}",
                #name,
                inner_error
            )
        };

        self.register_error(
            "inner_decoding_failure",
            variant_name,
            variant_type,
            display_match_arm,
        );
    }

    fn sequence_too_long(&self) -> TokenStream {
        let error = self.errors.get("seq_too_long").unwrap();
        self.global_identifier(&error.variant_name)
    }

    fn sequence_empty(&self) -> TokenStream {
        let error = self.errors.get("seq_empty").unwrap();
        self.global_identifier(&error.variant_name)
    }

    fn sequence_empty_for_field(&self) -> TokenStream {
        let error = self.errors.get("seq_empty_for_field").unwrap();
        self.global_identifier(&error.variant_name)
    }

    fn sequence_too_short_for_field(&self) -> TokenStream {
        let error = self.errors.get("seq_too_short_for_field").unwrap();
        self.global_identifier(&error.variant_name)
    }

    fn sequence_empty_for_variant(&self) -> TokenStream {
        let error = self.errors.get("seq_empty_for_variant").unwrap();
        self.global_identifier(&error.variant_name)
    }

    fn sequence_too_short_for_variant(&self) -> TokenStream {
        let error = self.errors.get("seq_too_short_for_variant").unwrap();
        self.global_identifier(&error.variant_name)
    }

    fn invalid_discriminant(&self) -> TokenStream {
        let error = self.errors.get("invalid_discriminant").unwrap();
        self.global_identifier(&error.variant_name)
    }

    fn into_tokens(self) -> TokenStream {
        let error_enum_name = self.error_enum_name();
        let inner_decoding_failure_name = self
            .errors
            .get("inner_decoding_failure")
            .unwrap()
            .variant_name
            .clone();

        let errors = self.errors.values();
        let variant_types = errors
            .clone()
            .map(|error| error.variant_type.clone())
            .collect::<Vec<_>>();
        let display_match_arms = errors
            .map(|error| error.display_match_arm.clone())
            .collect::<Vec<_>>();

        quote! {
            #[derive(::core::fmt::Debug)]
            pub enum #error_enum_name {
                #( #variant_types , )*
            }
            impl ::std::error::Error for #error_enum_name {}
            impl ::std::fmt::Display for #error_enum_name {
                fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                    match self {
                        #( #display_match_arms , )*
                    }
                }
            }
            impl ::core::convert::From<::std::boxed::Box<
                dyn ::std::error::Error + ::core::marker::Send + ::core::marker::Sync
            >>
            for #error_enum_name
            {
                fn from(err: ::std::boxed::Box<
                    dyn ::std::error::Error + ::core::marker::Send + ::core::marker::Sync
                >)
                -> Self {
                    Self::#inner_decoding_failure_name(err)
                }
            }
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use syn::parse_quote;

    use super::*;

    #[test]
    fn unit_struct() {
        let ast = parse_quote! {
            #[derive(BFieldCodec)]
            struct UnitStruct;
        };
        let _rust_code = BFieldCodecDeriveBuilder::new(ast).build();
    }

    #[test]
    fn tuple_struct() {
        let ast = parse_quote! {
            #[derive(BFieldCodec)]
            struct TupleStruct(u64, u32);
        };
        let _rust_code = BFieldCodecDeriveBuilder::new(ast).build();
    }

    #[test]
    fn struct_with_named_fields() {
        let ast = parse_quote! {
            #[derive(BFieldCodec)]
            struct StructWithNamedFields {
                field1: u64,
                field2: u32,
                #[bfield_codec(ignore)]
                ignored_field: bool,
            }
        };
        let _rust_code = BFieldCodecDeriveBuilder::new(ast).build();
    }

    #[test]
    fn enum_with_tuple_variants() {
        let ast = parse_quote! {
            #[derive(BFieldCodec)]
            enum Enum {
                Variant1,
                Variant2(u64),
                Variant3(u64, u32),
                #[bfield_codec(ignore)]
                IgnoredVariant,
            }
        };
        let _rust_code = BFieldCodecDeriveBuilder::new(ast).build();
    }

    #[test]
    fn generic_tuple_struct() {
        let ast = parse_quote! {
            #[derive(BFieldCodec)]
            struct TupleStruct<T>(T, (T, T));
        };
        let _rust_code = BFieldCodecDeriveBuilder::new(ast).build();
    }

    #[test]
    fn generic_struct_with_named_fields() {
        let ast = parse_quote! {
            #[derive(BFieldCodec)]
            struct StructWithNamedFields<T> {
                field1: T,
                field2: (T, T),
                #[bfield_codec(ignore)]
                ignored_field: bool,
            }
        };
        let _rust_code = BFieldCodecDeriveBuilder::new(ast).build();
    }

    #[test]
    fn generic_enum() {
        let ast = parse_quote! {
            #[derive(BFieldCodec)]
            enum Enum<T> {
                Variant1,
                Variant2(T),
                Variant3(T, T),
                #[bfield_codec(ignore)]
                IgnoredVariant,
            }
        };
        let _rust_code = BFieldCodecDeriveBuilder::new(ast).build();
    }
}
