extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;

#[proc_macro_derive(BFieldCodec)]
pub fn bfieldcodec_derive(input: TokenStream) -> TokenStream {
    // ...
    // Construct a representation of Rust code as a syntax tree
    // that we can manipulate
    let ast = syn::parse(input).unwrap();

    // Build the trait implementation
    impl_bfieldcodec_macro(&ast)
}

fn struct_with_named_fields(
    fields: &syn::FieldsNamed,
) -> (
    Vec<quote::__private::TokenStream>,
    Vec<quote::__private::TokenStream>,
    quote::__private::TokenStream,
) {
    let fields: Vec<_> = fields.named.iter().collect();
    let field_names = fields.iter().map(|field| &field.ident);

    let field_types = fields.iter().map(|field| &field.ty);

    let encode_statements: Vec<_> = field_names
        .clone()
        .map(|fname| {
            quote! {
                let mut #fname: Vec<BFieldElement> = self.#fname.encode();
                elements.push(BFieldElement::new(#fname.len() as u64));
                elements.append(&mut #fname);
            }
        })
        .collect();

    let decode_statements: Vec<_> = field_types
        .clone()
        .zip(field_names.clone())
        .map(|(ftype, fname)| {
            quote! {
                let (#fname, sequence) = decode_field_length_prepended::<#ftype>(&sequence)?;
            }
        })
        .collect();

    let value_constructor = quote! { Self { #(#field_names,)* } };

    (decode_statements, encode_statements, value_constructor)
}

fn struct_with_unnamed_fields(
    fields: &syn::FieldsUnnamed,
) -> (
    Vec<quote::__private::TokenStream>,
    Vec<quote::__private::TokenStream>,
    quote::__private::TokenStream,
) {
    let indices: Vec<_> = (0..fields.unnamed.len()).map(syn::Index::from).collect();
    let field_types = fields.unnamed.iter().map(|field| &field.ty);

    // Generate variables to capture decoded field values
    let field_names: Vec<_> = indices
        .iter()
        .map(|i| quote::format_ident!("field_value_{}", i.index))
        .collect();

    // Generate statements to decode each field
    let decode_statements: Vec<_> = field_types
        .clone()
        .zip(&field_names)
        .map(|(ty, var)| {
            let is_vec = if let syn::Type::Path(type_path) = ty {
                type_path.path.segments.last().unwrap().ident == "Vec"
            } else {
                false
            };

            if is_vec {
                quote! {
                    let (field_value, sequence) = decode_vec_length_prepended::<#ty>(&sequence)?;
                    let #var = field_value;
                }
            } else {
                quote! {
                    let (field_value, sequence) = decode_field_length_prepended::<#ty>(&sequence)?;
                    let #var = field_value;
                }
            }
        })
        .collect();

    let encode_statements: Vec<_> = indices
        .clone()
        .iter()
        .map(|idx| {
            quote! {
                let mut field_value: Vec<BFieldElement> = self.#idx.encode();
                elements.push(BFieldElement::new(field_value.len() as u64));
                elements.append(&mut field_value);
            }
        })
        .collect();

    let value_constructor: quote::__private::TokenStream = quote! { Self ( #(#field_names,)* ) };

    (decode_statements, encode_statements, value_constructor)
}

fn impl_bfieldcodec_macro(ast: &syn::DeriveInput) -> TokenStream {
    let (decode_statements, encode_statements, value_constructor) = match &ast.data {
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
    let gen = quote! {
        impl BFieldCodec for #name {
            fn decode(sequence: &[BFieldElement]) -> anyhow::Result<Box<Self>> {
                let mut sequence = sequence.to_vec();
                #(#decode_statements)*

                if !sequence.is_empty() {
                    anyhow::bail!("Failed to decode {}", stringify!(#name));
                }

                Ok(Box::new(#value_constructor))
            }

            fn encode(&self) -> Vec<BFieldElement> {
                let mut elements = Vec::new();
                #(#encode_statements)*
                elements
            }
        }
    };

    gen.into()
}
