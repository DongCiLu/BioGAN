# bio_gan
GAN modification for biology images (celegans microscope images)

We have some microscope images and we want to generate more samples like these with GAN.

Raw and denoised image of celegans slice:

![alt text](./examples/raw_image.jpg?raw=true "Raw Image of celegans slice")     ![alt text](./examples/denoised_image.jpg?raw=true "Denoised Image of celegans slice")


We use our dataset to train both unconditional GAN and conditional GAN, based on TFGAN library and get some preliminary results:

1. Unconditional GAN:

![alt text](./examples/unconditional_gan.png?raw=true "Results for unconditional GAN")

2. Conditional GAN:

![alt text](./examples/conditional_gan.png?raw=true "Results for conditional GAN")
