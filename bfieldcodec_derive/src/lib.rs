//! This crate provides a derive macro for the `BFieldCodec` trait.

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;

/// Derives `BFieldCodec` for structs.
///
/// Fields that should not be serialized can be ignored by annotating them with
/// `#[bfield_codec(ignore)]`.
/// Ignored fields must implement [`Default`].
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
/// ### Known limitations
/// ```
#[proc_macro_derive(BFieldCodec, attributes(bfield_codec))]
pub fn bfieldcodec_derive(input: TokenStream) -> TokenStream {
    // ...
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast = syn::parse(input).unwrap();

    // Build the trait implementation
    impl_bfieldcodec_macro(ast)
}
fn impl_bfieldcodec_macro(ast: syn::DeriveInput) -> TokenStream {
    let mut ignored_fields = vec![];
    let (encode_statements, decode_function_body, static_length_body) = match &ast.data {
        syn::Data::Struct(syn::DataStruct {
            fields: syn::Fields::Named(fields),
            ..
        }) => {
            ignored_fields = fields
                .named
                .iter()
                .filter(|field| field_is_ignored(field))
                .cloned()
                .collect::<Vec<_>>();
            generate_tokens_for_struct_with_named_fields(fields)
        }
        syn::Data::Struct(syn::DataStruct {
            fields: syn::Fields::Unnamed(fields),
            ..
        }) => generate_tokens_for_struct_with_unnamed_fields(fields),
        syn::Data::Enum(data_enum) => generate_tokens_for_enum(&data_enum.variants),
        _ => panic!("expected a struct with named fields, with unnamed fields, or an enum"),
    };

    let name = &ast.ident;

    // Extract all generics we shall ignore.
    let mut ignored = extract_ignored_generics_list(&ast.attrs);

    // For all ignored fields, add all type identifiers (including, recursively, the type
    // identifiers of generic type arguments) to the list of ignored type identifiers.
    let mut ignored_types: Vec<syn::Type> =
        ignored_fields.iter().map(|igf| igf.ty.clone()).collect();
    while !ignored_types.is_empty() {
        let ignored_type = ignored_types[0].clone();
        ignored_types = ignored_types[1..].to_vec();
        match ignored_type {
            syn::Type::Path(type_path) => {
                for segment in type_path.path.segments.into_iter() {
                    ignored.push(segment.ident);
                    match segment.arguments {
                        syn::PathArguments::AngleBracketed(angle_bracketed_generic_arguments) => {
                            for abga in angle_bracketed_generic_arguments.args.into_iter() {
                                match abga {
                                    syn::GenericArgument::Type(t) => {
                                        ignored_types.push(t.clone());
                                    }
                                    _ => continue,
                                }
                            }
                        }
                        _ => continue,
                    }
                }
            }
            _ => continue,
        }
    }

    // Add a bound `T: BFieldCodec` to every type parameter T, unless it is ignored
    let generics = add_trait_bounds(ast.generics, &ignored);

    // Extract the generics of the struct/enum.
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let gen = quote! {
        impl #impl_generics ::twenty_first::shared_math::bfield_codec::BFieldCodec
        for #name #ty_generics #where_clause{
            fn decode(
                sequence: &[::twenty_first::shared_math::b_field_element::BFieldElement],
            ) -> anyhow::Result<Box<Self>> {
                #decode_function_body
            }

            fn encode(&self) -> Vec<::twenty_first::shared_math::b_field_element::BFieldElement> {
                let mut elements = Vec::new();
                #(#encode_statements)*
                elements
            }

            fn static_length() -> Option<usize> {
                #static_length_body
            }
        }
    };

    gen.into()
}

/// Add a bound `T: BFieldCodec` to every type parameter T, unless we ignore it.
fn add_trait_bounds(mut generics: syn::Generics, ignored: &[syn::Ident]) -> syn::Generics {
    for param in &mut generics.params {
        let syn::GenericParam::Type(type_param) = param else {
            continue;
        };
        if ignored.contains(&type_param.ident) {
            continue;
        }
        type_param.bounds.push(syn::parse_quote!(BFieldCodec));
    }
    generics
}

fn extract_ignored_generics_list(list: &[syn::Attribute]) -> Vec<syn::Ident> {
    list.iter().flat_map(extract_ignored_generics).collect()
}

fn extract_ignored_generics(attr: &syn::Attribute) -> Vec<syn::Ident> {
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

fn field_is_ignored(field: &syn::Field) -> bool {
    for attribute in field.attrs.iter() {
        if !attribute.path().is_ident("bfield_codec") {
            continue;
        }
        attribute
            .parse_nested_meta(|meta| match meta.path.get_ident() {
                Some(ident) if ident == "ignore" => Ok(()),
                Some(ident) => Err(meta.error(format!("Unknown identifier \"{ident}\"."))),
                _ => Err(meta.error("Expected an identifier.")),
            })
            .unwrap();
        return true;
    }
    false
}

fn generate_tokens_for_struct_with_named_fields(
    fields: &syn::FieldsNamed,
) -> (
    Vec<quote::__private::TokenStream>,
    quote::__private::TokenStream,
    quote::__private::TokenStream,
) {
    let fields = fields.named.iter();
    let included_fields = fields.clone().filter(|field| !field_is_ignored(field));
    let ignored_fields = fields.clone().filter(|field| field_is_ignored(field));

    let included_field_names = included_fields
        .clone()
        .map(|field| field.ident.as_ref().unwrap().to_owned());
    let ignored_field_names = ignored_fields
        .clone()
        .map(|field| field.ident.as_ref().unwrap().to_owned());

    let included_field_types = included_fields.clone().map(|field| field.ty.clone());

    let encode_statements = included_field_names
        .clone()
        .zip(included_field_types.clone())
        .map(|(fname, field_type)| {
            quote! {
                let mut #fname: Vec<::twenty_first::shared_math::b_field_element::BFieldElement>
                    = self.#fname.encode();
                if <#field_type as ::twenty_first::shared_math::bfield_codec::BFieldCodec>
                    ::static_length().is_none() {
                    elements.push(
                        ::twenty_first::shared_math::b_field_element::BFieldElement::new(
                            #fname.len() as u64
                        )
                    );
                }
                elements.append(&mut #fname);
            }
        })
        .collect();

    let decode_statements = included_field_types
        .clone()
        .zip(included_field_names.clone())
        .map(|(ftype, fname)| generate_decode_statement_for_field(&fname, &ftype))
        .collect::<Vec<_>>();

    let value_constructor = quote! {
        Self {
            #(#included_field_names,)*
            #(#ignored_field_names: Default::default(),)*
        }
    };

    let decode_constructor = quote! {
        #(#decode_statements)*
        if !sequence.is_empty() {
            anyhow::bail!("Could not decode struct: sequence too long. ({} elements remaining)", sequence.len());
        }

        Ok(Box::new(#value_constructor))
    };

    (
        encode_statements,
        decode_constructor,
        generate_static_length_function_body_for_struct(included_field_types.collect()),
    )
}

fn generate_static_length_function_body_for_struct(
    field_types: Vec<syn::Type>,
) -> quote::__private::TokenStream {
    let num_fields = field_types.len();
    quote! {

        let field_lengths : [Option<usize>; #num_fields] = [
            #(
                <#field_types as
                ::twenty_first::shared_math::bfield_codec::BFieldCodec>::static_length(),
            )*
        ];
        if field_lengths.iter().all(|fl| fl.is_some() ) {
            Some(field_lengths.iter().map(|fl| fl.unwrap()).sum())
        }
        else {
            None
        }
    }
}

fn generate_tokens_for_struct_with_unnamed_fields(
    fields: &syn::FieldsUnnamed,
) -> (
    Vec<quote::__private::TokenStream>,
    quote::__private::TokenStream,
    quote::__private::TokenStream,
) {
    let indices: Vec<_> = (0..fields.unnamed.len()).map(syn::Index::from).collect();
    let field_types = fields
        .unnamed
        .iter()
        .map(|field| field.ty.clone())
        .collect::<Vec<_>>();

    // Generate variables to capture decoded field values
    let field_names: Vec<_> = indices
        .iter()
        .map(|i| quote::format_ident!("field_value_{}", i.index))
        .collect();

    // Generate statements to decode each field
    let decode_statements: Vec<_> = field_types
        .iter()
        .zip(&field_names)
        .map(|(ty, var)| generate_decode_statement_for_field(var, ty))
        .collect();

    let encode_statements: Vec<_> = indices
        .iter()
        .zip(field_types.clone())
        .map(|(idx, field_type)| {
            quote! {
                let mut field_value:
                    Vec<::twenty_first::shared_math::b_field_element::BFieldElement>
                    = self.#idx.encode();
                if <#field_type as ::twenty_first::shared_math::bfield_codec::BFieldCodec>
                    ::static_length().is_none() {
                    elements.push(::twenty_first::shared_math::b_field_element::BFieldElement::new(
                        field_value.len() as u64)
                    );
                }
                elements.append(&mut field_value);
            }
        })
        .collect();

    let value_constructor = quote! { Self ( #(#field_names,)* ) };

    let decode_constructor = quote! {
        #(#decode_statements)*
        if !sequence.is_empty() {
            anyhow::bail!("Could not decode struct: sequence too long. ({} elements remaining)", sequence.len());
        }
        Ok(Box::new(#value_constructor))
    };

    (
        encode_statements,
        decode_constructor,
        generate_static_length_function_body_for_struct(field_types),
    )
}

fn generate_tokens_for_enum(
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::token::Comma>,
) -> (
    Vec<quote::__private::TokenStream>,
    quote::__private::TokenStream,
    quote::__private::TokenStream,
) {
    let decode_clauses = variants
        .iter()
        .enumerate()
        .map(|(i, v)| generate_decode_clause_for_variant(i, &v.ident, &v.fields));
    let match_clauses = decode_clauses
        .enumerate()
        .map(|(i, c)| quote! { #i => { #c } });
    let decode_constructor = quote! {

        if sequence.is_empty() {
            anyhow::bail!("Cannot decode variant: sequence is empty");
        }
        else
        {
            let (variant_index, sequence) = (sequence[0].value() as usize, &sequence[1..]);
            match variant_index {
                #(#match_clauses ,)*
                other_index => anyhow::bail!("Cannot decode variant: invalid variant index: {other_index}"),
            }
        }

    };

    let encode_clauses = variants
        .iter()
        .enumerate()
        .map(|(i, v)| generate_encode_clause_for_variant(i, &v.ident, &v.fields));
    let encode_match_statement = quote! {
        match self {
            #( #encode_clauses , )*
        }
    };

    let num_variants = variants.len();
    let static_field_length_function_body = if variants.iter().all(|v| v.fields.is_empty()) {
        // no variants have associated data
        quote! {Some(1)}
    } else if num_variants == 0 {
        // there are no variants
        quote! {Some(0)}
    } else {
        // some variants have associated data
        // if all variants encode to the same length anyway, the length is still statically known
        let variant_lengths = variants
            .iter()
            .map(|variant| {
                let num_fields = variant.fields.len();
                let fields = variant.fields.clone();
                let field_lengths = fields
                    .iter()
                    .map(|f| quote!{
                        < #f as ::twenty_first::shared_math::bfield_codec::BFieldCodec>::static_length()
                    });
                quote!{
                    {
                        let field_lengths : [Option<usize>; #num_fields] = [ #( #field_lengths , )* ];
                        if field_lengths.iter().all(|fl| fl.is_some()) {
                            Some(field_lengths.iter().map(|fl|fl.unwrap()).sum())
                        } else {
                            None
                        }
                    }
                }
            })
            .collect::<Vec<_>>();
        quote! {

            let variant_lengths : [Option<usize>; #num_variants] = [
                #(
                    #variant_lengths ,
                )*
            ];
            if variant_lengths.iter().all(|fl| fl.is_some() ) && variant_lengths.iter().all(|x| x.unwrap() == variant_lengths[0].unwrap()) {
                variant_lengths[0]
            }
            else {
                None
            }
        }
    };

    (
        vec![encode_match_statement],
        decode_constructor,
        static_field_length_function_body,
    )
}

fn generate_decode_statement_for_field(
    field_name: &syn::Ident,
    field_type: &syn::Type,
) -> quote::__private::TokenStream {
    let field_name_as_string_literal = field_name.to_string();
    quote! {
        let (#field_name, sequence) = {
            let maybe_fields_static_length = <#field_type as ::twenty_first::shared_math::bfield_codec::BFieldCodec>::static_length();
            let field_has_dynamic_length = maybe_fields_static_length.is_none();
            if sequence.is_empty() && field_has_dynamic_length {
                anyhow::bail!("Cannot decode field {}: sequence is empty.", #field_name_as_string_literal);
            }
            let (len, sequence) = match maybe_fields_static_length {
                Some(len) => (len, sequence),
                None => (sequence[0].value() as usize, &sequence[1..]),
            };
            if sequence.len() < len {
                anyhow::bail!("Cannot decode field {}: sequence too short.", #field_name_as_string_literal);
            }
            let decoded = *<#field_type as ::twenty_first::shared_math::bfield_codec::BFieldCodec>::decode(
                    &sequence[..len]
                )?;
            (decoded, &sequence[len..])
        };
    }
}

fn enum_variant_field_name(variant_index: usize, field_index: usize) -> syn::Ident {
    quote::format_ident!("variant_{}_field_{}", variant_index, field_index)
}

fn generate_decode_clause_for_variant(
    variant_index: usize,
    name: &syn::Ident,
    associated_data: &syn::Fields,
) -> quote::__private::TokenStream {
    if associated_data.is_empty() {
        quote! {
            if !sequence.is_empty() {
                anyhow::bail!("Cannot decode enum: sequence too long.");
            }
            Ok(Box::new(Self::#name))
        }
    } else {
        let field_decoders = associated_data.iter().enumerate().map(|(field_index, field)| {
            let field_type = field.ty.clone();
            let field_name = enum_variant_field_name(variant_index, field_index);
            let field_value = quote::format_ident!("variant_{}_field_{}_value", variant_index, field_index);
            quote! {
                let (#field_value, sequence) = {
                    let maybe_fields_static_length = <#field_type as ::twenty_first::shared_math::bfield_codec::BFieldCodec>::static_length();
                    let field_has_dynamic_length = maybe_fields_static_length.is_none();
                    if sequence.is_empty() && field_has_dynamic_length {
                        anyhow::bail!("Cannot decode variant {} field {}: sequence is empty.", #variant_index, #field_index);
                    }
                    let (len, sequence) = match maybe_fields_static_length {
                        Some(len) => (len, sequence),
                        None => (sequence[0].value() as usize, &sequence[1..]),
                    };
                    if sequence.len() < len {
                        anyhow::bail!("Cannot decode variant {} field {}: sequence too short.", #variant_index, #field_index);
                    }
                    let decoded = *<#field_type as ::twenty_first::shared_math::bfield_codec::BFieldCodec>::decode(
                            &sequence[..len]
                        )?;
                    (decoded, &sequence[len..])
                };
                let #field_name = #field_value;
            }
        }).fold(quote!{}, |l, r| quote!{#l #r});
        let field_names = associated_data
            .iter()
            .enumerate()
            .map(|(field_index, _field)| enum_variant_field_name(variant_index, field_index));
        quote! {
            #field_decoders
            if !sequence.is_empty() {
                anyhow::bail!("Cannot decode enum: sequence too long.");
            }
            Ok(Box::new(Self::#name ( #( #field_names , )* )))
        }
    }
}

fn generate_encode_clause_for_variant(
    variant_index: usize,
    variant_name: &syn::Ident,
    associated_data: &syn::Fields,
) -> quote::__private::TokenStream {
    if associated_data.is_empty() {
        quote! {
            Self::#variant_name => { elements.push(::twenty_first::shared_math::b_field_element::BFieldElement::new( #variant_index as u64)); }
        }
    } else {
        let field_encoders = associated_data.iter().enumerate().map(|(field_index, ad)| {
            let field_name = enum_variant_field_name(variant_index, field_index);
            let field_type = ad.ty.clone();
            let field_encoding =
                quote::format_ident!("variant_{}_field_{}_encoding", variant_index, field_index);
            quote! {
                let mut #field_encoding :
                    Vec<::twenty_first::shared_math::b_field_element::BFieldElement>
                    = #field_name.encode();
                if <#field_type as ::twenty_first::shared_math::bfield_codec::BFieldCodec>
                    ::static_length().is_none() {
                    elements.push(::twenty_first::shared_math::b_field_element::BFieldElement::new(
                        #field_encoding.len() as u64)
                    );
                }
                elements.append(&mut #field_encoding);
            }
        });

        let field_names = associated_data
            .iter()
            .enumerate()
            .map(|(field_index, _field)| enum_variant_field_name(variant_index, field_index));

        let ret = quote! {
            Self::#variant_name ( #( #field_names , )* ) => {
                elements.push(::twenty_first::shared_math::b_field_element::BFieldElement::new( #variant_index as u64));
                #( #field_encoders )*
            }
        };

        ret
    }
}
