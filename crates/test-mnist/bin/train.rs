#![recursion_limit = "256"]

use test_mnist::{train_device, training};

fn main() {
    training::run(train_device());
}
