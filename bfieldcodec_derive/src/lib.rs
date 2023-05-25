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

fn impl_bfieldcodec_macro(ast: &syn::DeriveInput) -> TokenStream {
    fn struct_with_named_fields(name: &syn::Ident, fields: &syn::FieldsNamed) -> TokenStream {
        let fields: Vec<_> = fields.named.iter().collect();
        let field_name = fields.iter().map(|field| &field.ident);

        let field_name2 = fields.iter().map(|field| &field.ident);
        let field_name3 = fields.iter().map(|field| &field.ident);
        let field_type = fields.iter().map(|field| &field.ty);
        let gen = quote! {
            impl BFieldCodec for #name {
                fn decode(sequence: &[BFieldElement]) -> anyhow::Result<Box<Self>> {
                    let mut sequence = sequence.to_vec();
                    #(
                        let (#field_name2, sequence) = decode_field_length_prepended::<#field_type>(&sequence)?;
                    )*

                    if !sequence.is_empty() {
                        anyhow::bail!("Failed to decode {}", stringify!(#name));
                    }

                    Ok(Box::new(Self { #(#field_name3,)* }))
                }

                fn encode(&self) -> Vec<BFieldElement> {
                    let mut elements = Vec::new();
                    #(
                        let mut #field_name: Vec<BFieldElement> = self.#field_name.encode();
                        elements.push(BFieldElement::new(#field_name.len() as u64));
                        elements.append(&mut #field_name);
                    )*
                    elements
                }
            }
        };

        gen.into()
    }

    fn struct_with_unnamed_fields(name: &syn::Ident, fields: &syn::FieldsUnnamed) -> TokenStream {
        let indices: Vec<_> = (0..fields.unnamed.len()).map(syn::Index::from).collect();

        let indices2 = indices.clone();
        let field_type = fields.unnamed.iter().map(|field| &field.ty);

        // Generate variables to capture decoded field values
        let field_vars: Vec<_> = indices
            .iter()
            .map(|i| quote::format_ident!("field_value_{}", i.index))
            .collect();

        // Generate statements to decode each field
        let decode_statements: Vec<_> = field_type
            .clone()
            .zip(&field_vars)
            .map(|(ty, var)| {
                quote! {
                    let (field_value, sequence) = decode_field_length_prepended::<#ty>(&sequence)?;
                    let #var = field_value;
                }
            })
            .collect();

        let gen = quote! {
            impl BFieldCodec for #name {
                fn decode(sequence: &[BFieldElement]) -> anyhow::Result<Box<Self>> {
                    let mut sequence = sequence.to_vec();
                    #(#decode_statements)*

                    if !sequence.is_empty() {
                        anyhow::bail!("Failed to decode {}", stringify!(#name));
                    }

                    let res = Self( #(#field_vars),* );

                    Ok(Box::new(res))
                }

                fn encode(&self) -> Vec<BFieldElement> {
                    let mut elements = Vec::new();
                    #(
                        let mut field_value: Vec<BFieldElement> = self.#indices2.encode();
                        elements.push(BFieldElement::new(field_value.len() as u64));
                        elements.append(&mut field_value);
                    )*
                    elements
                }
            }
        };

        gen.into()
    }

    let name = &ast.ident;
    match &ast.data {
        syn::Data::Struct(syn::DataStruct {
            fields: syn::Fields::Named(fields),
            ..
        }) => struct_with_named_fields(name, fields),
        syn::Data::Struct(syn::DataStruct {
            fields: syn::Fields::Unnamed(fields),
            ..
        }) => struct_with_unnamed_fields(name, fields),
        _ => panic!("expected a struct with named fields"),
    }
}
