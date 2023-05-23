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
    let name = &ast.ident;
    let fields: Vec<_> = match &ast.data {
        syn::Data::Struct(syn::DataStruct {
            fields: syn::Fields::Named(fields),
            ..
        }) => fields.named.iter().collect(),
        _ => panic!("expected a struct with named fields"),
    };
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
