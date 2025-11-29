import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def generate_font_data(samples_per_digit=500):
    """
    Generates synthetic data using computer fonts to help the model 
    recognize printed Sudoku numbers, including colored backgrounds.
    """
    print(f"Generating {samples_per_digit * 10} synthetic font images (with colored backgrounds)...")
    
    # OpenCV Fonts to simulate different printed styles
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    ]
    
    data = []
    labels = []
    
    for digit in range(10):
        for _ in range(samples_per_digit):
            # 1. Create image with random background color
            # Simulate white, light gray, and light blue backgrounds (like sudoku.com)
            bg_type = np.random.choice(['white', 'light_gray', 'light_blue', 'very_light_blue'])
            
            if bg_type == 'white':
                bg_color = np.random.randint(240, 256)
            elif bg_type == 'light_gray':
                bg_color = np.random.randint(220, 245)
            elif bg_type == 'light_blue':
                bg_color = np.random.randint(200, 235)
            else:  # very_light_blue
                bg_color = np.random.randint(210, 245)
            
            img = np.full((28, 28), bg_color, dtype=np.uint8)
            
            # 2. Randomly choose a font, scale, and thickness
            font = np.random.choice(fonts)
            scale = np.random.uniform(0.5, 1.2)
            thickness = np.random.randint(1, 3)
            
            # 3. Calculate text size to center it
            text = str(digit)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
            
            x = (28 - text_width) // 2
            y = (28 + text_height) // 2
            
            # Add some random shift
            x += np.random.randint(-3, 4)
            y += np.random.randint(-3, 4)
            
            # 4. Choose text color (dark blue/black like sudoku.com)
            text_color_type = np.random.choice(['black', 'dark_gray', 'dark_blue'])
            if text_color_type == 'black':
                text_color = np.random.randint(0, 40)
            elif text_color_type == 'dark_gray':
                text_color = np.random.randint(30, 80)
            else:  # dark_blue
                text_color = np.random.randint(40, 100)
            
            # 5. Draw the digit
            cv2.putText(img, text, (x, y), font, scale, (text_color), thickness, cv2.LINE_AA)
            
            # 6. Add random transformations
            # Random rotation
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
                img = cv2.warpAffine(img, M, (28, 28), borderValue=bg_color)
            
            # 7. Add random noise (simulating camera artifacts)
            if np.random.random() > 0.3:
                noise = np.random.randint(-20, 20, (28, 28), dtype=np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # 8. Random blur (simulating out-of-focus)
            if np.random.random() > 0.5:
                kernel_size = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
            # 9. Add grid line artifacts (simulating partial grid lines)
            if np.random.random() > 0.6:
                # Add a line segment on one edge
                edge = np.random.choice(['top', 'bottom', 'left', 'right'])
                line_color = np.random.randint(50, 150)
                thickness = np.random.randint(1, 3)
                
                if edge == 'top':
                    img[0:thickness, :] = line_color
                elif edge == 'bottom':
                    img[-thickness:, :] = line_color
                elif edge == 'left':
                    img[:, 0:thickness] = line_color
                else:  # right
                    img[:, -thickness:] = line_color
            
            # 10. Normalize and add to list
            data.append(img)
            labels.append(digit)
            
    # Convert to numpy arrays
    data = np.array(data, dtype='float32')
    data = data.reshape((data.shape[0], 28, 28, 1)) / 255.0
    labels = to_categorical(labels, 10)
    
    return data, labels

def train_hybrid_model():
    # 1. Load MNIST Data (Handwriting)
    print("Loading MNIST Data...")
    (x_train_mnist, y_train_mnist), (x_test, y_test) = mnist.load_data()
    
    # Preprocess MNIST (keep original, our extract_digit will handle it)
    x_train_mnist = x_train_mnist.reshape((x_train_mnist.shape[0], 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
    y_train_mnist = to_categorical(y_train_mnist, 10)
    y_test = to_categorical(y_test, 10)
    
    # 2. Generate Font Data (Computer Text with colored backgrounds)
    # We generate 2000 samples per digit = 20,000 images (focused on printed digits)
    x_train_fonts, y_train_fonts = generate_font_data(samples_per_digit=2000)
    
    # 3. Combine Datasets
    print("Merging Datasets...")
    x_train = np.concatenate((x_train_mnist, x_train_fonts), axis=0)
    y_train = np.concatenate((y_train_mnist, y_train_fonts), axis=0)
    
    # Shuffle the data
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    print(f"Total training samples: {x_train.shape[0]}")

    # 4. Build Model (simpler but effective)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. Train with Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1
    )
    datagen.fit(x_train)

    print("Training Model for Sudoku.com (colored backgrounds)...")
    print(f"Total training samples: {x_train.shape[0]}")
    
    model.fit(datagen.flow(x_train, y_train, batch_size=32),
              epochs=7,
              validation_data=(x_test, y_test))

    # 6. Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    
    # 7. Save
    model.save('digit_model.h5')
    print("\n✓ Success! Enhanced model saved as 'digit_model.h5'")
    print("This model is optimized for sudoku.com style images with colored backgrounds!")

if __name__ == "__main__":
    train_hybrid_model()