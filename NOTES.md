## notes

1. since we are doing this in rust, we could leverage the rich type system of
   rust and make different tensor dimensionalities as a type and that would
   give us compile time guarantees for different tensor ops. but i find that
   making dimensionalities as a type would only make things more convoluted for
   implementing different tensor ops like broadcasting and also i don't think
   it aligns with the design principle of this crate, which tries to be minimal
   and performance is the main goal of this crate. we want to be at least
   on-par with `ggml` on cpu.

2. data of a tensor should be mutable by default, right now we wrap the data
   inside `Arc` which does not allow us to ergonomically borrow the data
   mutably without the use of `RefCell`'s. since we only clone the underlying
   data when modifying the shape, we can change all shape op methods to mutably
   change the shape of the tensor and not produce `Arc` cloned tensors.
