extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;

#[proc_macro_derive(BFieldCodec, attributes(bfield_codec))]
pub fn bfieldcodec_derive(input: TokenStream) -> TokenStream {
    // ...
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast = syn::parse(input).unwrap();

    // Build the trait implementation
    impl_bfieldcodec_macro(ast)
}

// Add a bound `T: BFieldCodec` to every type parameter T, unless we ignore it.
fn add_trait_bounds(mut generics: syn::Generics, ignored: &[String]) -> syn::Generics {
    for param in &mut generics.params {
        if let syn::GenericParam::Type(type_param) = param {
            let name = type_param.ident.to_string();
            let mut found = false;
            for ignored in ignored.iter() {
                if ignored == &name {
                    found = true;
                    break;
                }
            }
            if found {
                continue;
            }
            type_param.bounds.push(syn::parse_quote!(BFieldCodec));
        }
    }
    generics
}

fn extract_ignored_generics_list(list: &[syn::Attribute]) -> Vec<String> {
    let mut collection = Vec::new();

    for attr in list.iter() {
        let mut list = extract_ignored_generics(attr);

        collection.append(&mut list);
    }

    collection
}

fn extract_ignored_generics(attr: &syn::Attribute) -> Vec<String> {
    let mut collection = Vec::new();

    if let Ok(meta) = attr.parse_meta() {
        if let Some(ident) = meta.path().get_ident() {
            if &ident.to_string() != "bfield_codec" {
                return collection;
            }
            if let syn::Meta::List(list) = meta {
                for nested in list.nested.iter() {
                    if let syn::NestedMeta::Meta(nmeta) = nested {
                        let ident = nmeta
                            .path()
                            .get_ident()
                            .expect("Invalid attribute syntax! (no iden)");
                        if &ident.to_string() != "ignore" {
                            panic!(
                                "Invalid attribute syntax! Unknown name {:?}",
                                ident.to_string()
                            );
                        }

                        if let syn::Meta::List(list) = nmeta {
                            for nested in list.nested.iter() {
                                if let syn::NestedMeta::Meta(syn::Meta::Path(path)) = nested {
                                    let path = path
                                        .get_ident()
                                        .expect("Invalid attribute syntax! (no ident)")
                                        .to_string();
                                    collection.push(path);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    collection
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
        impl #impl_generics BFieldCodec for #name #ty_generics #where_clause{
            fn decode(sequence: &[crate::shared_math::b_field_element::BFieldElement]) -> anyhow::Result<Box<Self>> {
                let mut sequence = sequence.to_vec();
                #(#decode_statements)*

                if !sequence.is_empty() {
                    anyhow::bail!("Failed to decode {}", stringify!(#name));
                }

                Ok(Box::new(#value_constructor))
            }

            fn encode(&self) -> Vec<crate::shared_math::b_field_element::BFieldElement> {
                let mut elements = Vec::new();
                #(#encode_statements)*
                elements
            }

            fn static_length() -> Option<usize> {
                let field_lengths : [Option<usize>; #num_fields] = [#(<#field_types as BFieldCodec>::static_length(),)*];
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

fn struct_with_named_fields(
    fields: &syn::FieldsNamed,
) -> (
    Vec<quote::__private::TokenStream>,
    Vec<quote::__private::TokenStream>,
    quote::__private::TokenStream,
    Vec<syn::Type>,
) {
    let fields: Vec<_> = fields.named.iter().collect();
    let field_names: Vec<_> = fields
        .iter()
        .map(|field| field.ident.as_ref().unwrap().to_owned())
        .collect();

    let field_types = fields
        .iter()
        .map(|field| field.ty.clone())
        .collect::<Vec<_>>();

    let encode_statements: Vec<_> = field_names
        .iter()
        .map(|fname| {
            quote! {
                let mut #fname: Vec<crate::shared_math::b_field_element::BFieldElement> = self.#fname.encode();
                elements.push(crate::shared_math::b_field_element::BFieldElement::new(#fname.len() as u64));
                elements.append(&mut #fname);
            }
        })
        .collect();

    let decode_statements: Vec<_> = field_types
        .iter()
        .zip(&field_names)
        .map(|(ftype, fname)| generate_decode_statement(fname, ftype))
        .collect();

    let value_constructor = quote! { Self { #(#field_names,)* } };

    (
        decode_statements,
        encode_statements,
        value_constructor,
        field_types,
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
        .map(|idx| {
            quote! {
                    let mut field_value: Vec<crate::shared_math::b_field_element::BFieldElement> = self.#idx.encode();
                    elements.push(crate::shared_math::b_field_element::BFieldElement::new(field_value.len() as u64));
                    elements.append(&mut field_value);
            }
        })
        .collect();

    let value_constructor: quote::__private::TokenStream = quote! { Self ( #(#field_names,)* ) };

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
        let (field_value, sequence) = {if sequence.is_empty() {
            anyhow::bail!("Cannot decode field: sequence is empty.");
        }
        let len = sequence[0].value() as usize;
        if sequence.len() < 1 + len {
            anyhow::bail!("Cannot decode field: sequence too short.");
        }
        let decoded = *<#field_type as BFieldCodec>::decode(&sequence[1..1 + len])?;
        (decoded, sequence[1 + len..].to_vec())};
        let #field_name = field_value;
    }
}
