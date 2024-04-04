import cv2
import numpy as np

img = cv2.imread("./images/1_0002.jpg", cv2.IMREAD_GRAYSCALE)

def visualize_contours(image, contours):
    img_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(img_with_contours, (center_x, center_y), 2, (255, 255, 255), -1)
        cv2.line(img_with_contours, (x + w // 2, y), (x + w // 2, y + h), (255, 0, 0), 1)  # Вертикальная линия высоты
        cv2.line(img_with_contours, (x, y + h // 2), (x + w, y + h // 2), (255, 0, 0), 1)  # Горизонтальная линия ширины
        cv2.putText(img_with_contours, "Capture Window", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Image with Contours", img_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_contours(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

contours = find_contours(img)

print(f"Количество обнаруженных фигур: {len(contours)}")

visualize_contours(img, contours)

def euclidean_distance(center1, center2):
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

areas = []
centers = []

for contour in contours:
    area = cv2.contourArea(contour)
    areas.append(area)
    x, y, w, h = cv2.boundingRect(contour)
    center_x = x + w // 2
    center_y = y + h // 2
    centers.append((center_x, center_y))

print("Площади и центры обнаруженных объектов:")
print("Площадь | Центр (x, y)")
print("-----------------------")
for i in range(len(contours)):
    print(f"{areas[i]:.2f} | {centers[i]}")

min_area_index = np.argmin(areas)
max_area_index = np.argmax(areas)
min_area = areas[min_area_index]
max_area = areas[max_area_index]
min_area_center = centers[min_area_index]
max_area_center = centers[max_area_index]

print(f"\nМинимальная площадь: {min_area:.2f}, центр: {min_area_center}")
print(f"Максимальная площадь: {max_area:.2f}, центр: {max_area_center}")

print("\nЕвклидовы расстояния между центрами объектов:")
print("---------------------------------------------")
for i in range(len(centers)):
    for j in range(i + 1, len(centers)):
        distance = euclidean_distance(centers[i], centers[j])
        print(f"Расстояние между центром {i+1} и центром {j+1}: {distance:.2f}")
