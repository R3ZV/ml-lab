import numpy as np
import matplotlib.pyplot as plt

def main():
    NUM_IMAGES = 9
    images = np.array([np.load(f"images/car_{i}.npy") for i in range(NUM_IMAGES)])
    print(images[0])

    images_sum = np.sum(images)

    print(f"Img sum: {images_sum}")
    pixels_sum = np.array([np.sum(image) for image in images])
    print(f"Images pixel sum: {pixels_sum}")
    print("Indexul sumei maxime: {}".format(np.argmax(pixels_sum)))

    mean_image = np.mean(images, axis=0)

    plt.imshow(mean_image.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.savefig("mean_image.png", bbox_inches='tight')

    standard_dev = np.std(images)
    normal_images = (images - mean_image) / standard_dev
    print(f"Normal images: {normal_images}")

    plt.imshow(normal_images[0].astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.savefig("first_normal.png", bbox_inches='tight')

    croped_images = images[:, 200:300, 280:400]


    plt.imshow(croped_images[0].astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.savefig("first_croped.png", bbox_inches='tight')

main()
