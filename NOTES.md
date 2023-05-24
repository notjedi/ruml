## notes

1. since we are doing this in rust, we could leverage the rich type system of
   rust and make tensor shapes as a type and that would give us compile time
   guarantees for different tensor ops. but i find that making shapes as a type
   would only make things more convoluted for implementing different tensor ops
   and also i don't think it aligns with the design principle of this crate,
   which tries to be as minimal as possible and performance is the main motive
   of this crate. we want to be on-par with ggml.

2. data of a tensor by default should be mutable, right now we wrap the data
   inside of `Arc` which does not allow us to ergonomically borrow the data
   mutably. since we only clone the underlying data when modifying the shape,
   we can change all shape op methods to mutably change the shape of the tensor
   and not produce Arc cloned tensors.
