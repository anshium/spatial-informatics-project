import cv2
import numpy as np
import matplotlib.pyplot as plt

def threshold_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    threshold_value = 0
    _, binary_image_manual = cv2.threshold(image, threshold_value, 255, cv2.THRESH_OTSU)
    return binary_image_manual

def plot_images(images, titles):
    plt.figure(figsize=(7, 7))
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def hough(image):
    regions = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=2.2,
        minDist=40,
        param1=600,
        param2=60,
        minRadius=5,
        maxRadius=380
    )
    
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    if regions is not None:
        regions = np.uint16(np.around(regions))
        num_coins = regions.shape[1]
        print(f"\nThe number of circles is: {num_coins}\n")

        for i in range(num_coins):
            center = (regions[0, i, 0], regions[0, i, 1])
            radius = regions[0, i, 2]
            
            cv2.circle(output_image, center, 3, (0, 255, 0), -1)
            cv2.circle(output_image, center, radius, (0, 0, 255), 3)
            print(f"Center location for circle {i + 1}: {center}\nDiameter: {2 * radius}\n")
    else:
        print("No circles were detected.")
    
    return output_image

def main():
    image_path = "datasets/CRISM/ice_deposits.png"
    
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    thresholded_image = threshold_image(image_path)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 93))  # Adjust kernel size as needed
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

    hough_image = hough(opened_image)

    plot_images(
        [original_image, thresholded_image, opened_image, hough_image],
        ['Original Image', 'Thresholded Image', 'Dilated Image', 'Hough Transform Output']
    )

if __name__ == "__main__":
    main()