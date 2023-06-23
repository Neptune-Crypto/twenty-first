//! This crate provides a derive macro for the `BFieldCodec` trait.

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::Ident;

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

/// Add a bound `T: BFieldCodec` to every type parameter T, unless we ignore it.
fn add_trait_bounds(mut generics: syn::Generics, ignored: &[Ident]) -> syn::Generics {
    for param in &mut generics.params {
        let syn::GenericParam::Type(type_param) = param else {
            continue
        };
        if ignored.contains(&type_param.ident) {
            continue;
        }
        type_param.bounds.push(syn::parse_quote!(BFieldCodec));
    }
    generics
}

fn extract_ignored_generics_list(list: &[syn::Attribute]) -> Vec<Ident> {
    list.iter().flat_map(extract_ignored_generics).collect()
}

fn extract_ignored_generics(attr: &syn::Attribute) -> Vec<Ident> {
    let bfield_codec_ident = Ident::new("bfield_codec", attr.span());
    let ignore_ident = Ident::new("ignore", attr.span());

    let Ok(meta) = attr.parse_meta() else {
        return vec![];
    };
    let Some(ident) = meta.path().get_ident() else {
        return vec![];
    };
    if ident != &bfield_codec_ident {
        return vec![];
    }
    let syn::Meta::List(list) = meta else {
        return vec![];
    };

    let mut ignored_generics = vec![];
    for nested in list.nested.iter() {
        let syn::NestedMeta::Meta(nmeta) = nested else {
            continue;
        };
        let Some(ident) = nmeta.path().get_ident() else {
            panic!("Invalid attribute syntax! (no ident)");
        };
        if ident != &ignore_ident {
            panic!("Invalid attribute syntax! Unknown name {ident}");
        }
        let syn::Meta::List(list) = nmeta else {
            panic!("Invalid attribute syntax! Expected a list");
        };

        for nested in list.nested.iter() {
            let syn::NestedMeta::Meta(syn::Meta::Path(path)) = nested else {
                continue;
            };
            let Some(ident) = path.get_ident() else {
                panic!("Invalid attribute syntax! (no ident)")
            };
            ignored_generics.push(ident.to_owned());
        }
    }
    ignored_generics
}

fn impl_bfieldcodec_macro(ast: syn::DeriveInput) -> TokenStream {
    let (decode_statements, encode_statements, value_constructor, field_types) = match &ast.data {
        syn::Data::Struct(syn::DataStruct {
            fields: syn::Fields::Named(fields),
            ..
        }) => struct_with_named_fields(fields),
        syn::Data::Struct(syn::DataStruct {
            fields: syn::Fields::Unnamed(fields),
            ..
        }) => struct_with_unnamed_fields(fields),
        _ => panic!("expected a struct with named fields"),
    };

    let name = &ast.ident;

    // Extract all generics we shall ignore.
    let ignored = extract_ignored_generics_list(&ast.attrs);

    // Add a bound `T: BFieldCodec` to every type parameter T.
    let generics = add_trait_bounds(ast.generics, &ignored);

    // Extract the generics of the struct/enum.
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let num_fields = field_types.len();

    let gen = quote! {
        impl #impl_generics ::twenty_first::shared_math::bfield_codec::BFieldCodec
        for #name #ty_generics #where_clause{
            fn decode(
                sequence: &[::twenty_first::shared_math::b_field_element::BFieldElement],
            ) -> anyhow::Result<Box<Self>> {
                #(#decode_statements)*
                if !sequence.is_empty() {
                    anyhow::bail!("Failed to decode {}", stringify!(#name));
                }
                Ok(Box::new(#value_constructor))
            }

            fn encode(&self) -> Vec<::twenty_first::shared_math::b_field_element::BFieldElement> {
                let mut elements = Vec::new();
                #(#encode_statements)*
                elements
            }

            fn static_length() -> Option<usize> {
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
    };

    gen.into()
}

fn field_is_ignored(field: &syn::Field) -> bool {
    let bfield_codec_ident = Ident::new("bfield_codec", field.span());
    let ignore_ident = Ident::new("ignore", field.span());

    for attribute in field.attrs.iter() {
        let Ok(meta) = attribute.parse_meta() else {
            continue;
        };
        let Some(ident) = meta.path().get_ident() else {
            continue;
        };
        if ident != &bfield_codec_ident {
            continue;
        }
        let syn::Meta::List(list) = meta else {
            panic!("Attribute {ident} must be of type `List`.");
        };
        for arg in list.nested.iter() {
            let syn::NestedMeta::Meta(arg_meta) = arg else {
                continue;
            };
            let Some(arg_ident) = arg_meta.path().get_ident() else {
                panic!("Invalid attribute syntax! (no ident)");
            };
            if arg_ident != &ignore_ident {
                panic!("Invalid attribute syntax! Unknown name {arg_ident}");
            }
            return true;
        }
    }
    false
}

fn struct_with_named_fields(
    fields: &syn::FieldsNamed,
) -> (
    Vec<quote::__private::TokenStream>,
    Vec<quote::__private::TokenStream>,
    quote::__private::TokenStream,
    Vec<syn::Type>,
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

    let decode_statements: Vec<_> = included_field_types
        .clone()
        .zip(included_field_names.clone())
        .map(|(ftype, fname)| generate_decode_statement(&fname, &ftype))
        .collect();

    let value_constructor = quote! {
        Self {
            #(#included_field_names,)*
            #(#ignored_field_names: Default::default(),)*
        }
    };

    (
        decode_statements,
        encode_statements,
        value_constructor,
        included_field_types.collect(),
    )
}

fn struct_with_unnamed_fields(
    fields: &syn::FieldsUnnamed,
) -> (
    Vec<quote::__private::TokenStream>,
    Vec<quote::__private::TokenStream>,
    quote::__private::TokenStream,
    Vec<syn::Type>,
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
        .map(|(ty, var)| generate_decode_statement(var, ty))
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

    (
        decode_statements,
        encode_statements,
        value_constructor,
        field_types,
    )
}

fn generate_decode_statement(
    field_name: &syn::Ident,
    field_type: &syn::Type,
) -> quote::__private::TokenStream {
    quote! {
        let (field_value, sequence) = {
            if sequence.is_empty() {
                anyhow::bail!("Cannot decode field: sequence is empty.");
            }
            let (len, sequence) = match <#field_type
                as ::twenty_first::shared_math::bfield_codec::BFieldCodec>::static_length() {
                Some(len) => (len, sequence),
                None => (sequence[0].value() as usize, &sequence[1..]),
            };
            if sequence.len() < len {
                anyhow::bail!("Cannot decode field: sequence too short.");
            }
            let decoded = *<#field_type
                as ::twenty_first::shared_math::bfield_codec::BFieldCodec>::decode(
                    &sequence[..len]
                )?;
            (decoded, &sequence[len..])
        };
        let #field_name = field_value;
    }
}
