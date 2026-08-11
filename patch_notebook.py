import json
from pathlib import Path

path = Path('RoadDamage.ipynb')
if not path.exists():
    raise SystemExit('Notebook file not found: ' + str(path))

with path.open('r', encoding='utf-8') as f:
    data = json.load(f)

changed = False

for cell in data.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    source = ''.join(cell.get('source', []))
    if source.startswith('#load damage road dataset'):
        cell['source'] = [
            '#load damage road dataset\n',
            "labels = ['D00', 'D10', 'D20', 'D40', 'Repair']\n",
            'target_size = (128, 128)\n',
            'if os.path.exists("model/X1.npy"):#if images already process then load it \n',
            "    data = np.load('model/X1.npy')\n",
            "    labels = np.load('model/Y1.npy')\n",
            "    bboxes = np.load('model/Z1.npy')\n",
            "    if data.dtype != np.float32 and data.dtype != np.float64:\n",
            "        data = data.astype('float32') / 255.0\n",
            'else: #if not process then read and save all images for training\n',
            '    X = []\n',
            '    Y = []\n',
            '    bb = []\n',
            '    path = "RDD2022_China_Drone/annotations"\n',
            '    for roots, dirs, directory in os.walk(path):#connect to dataset and loop all annotation and images\n',
            '        for j in range(len(directory)):\n',
            '            tree = ET.parse(roots+"/"+directory[j])#parse xml file to read bounding boxes annotation\n',
            '            root = tree.getroot()\n',
            '            img_name = root.find("filename").text\n',
            '            arr = img_name.split("_")\n',
            '            img = cv2.imread("RDD2022_China_Drone/images/"+img_name)#read image\n',
            '            if img is not None:\n',
            '                img = cv2.resize(img, target_size)\n',
            '                img = img.astype("float32") / 255.0\n',
            '                height, width, channel = img.shape\n',
            '                boxes = getBoxes()\n',
            '                index = 0\n',
            '                for item in root.findall("object"): #get boxes\n',
            '                    name = item.find("name").text\n',
            '                    xmin = float(item.find("bndbox/xmin").text)\n',
            '                    ymin = float(item.find("bndbox/ymin").text)\n',
            '                    xmax = float(item.find("bndbox/xmax").text)\n',
            '                    ymax = float(item.find("bndbox/ymax").text)\n',
            '                    if index < 12:\n',
            '                        xmin, ymin, xmax, ymax = normalizeBoxes([xmin, ymin, xmax, ymax], width, height)#normalize boxes\n',
            '                        boxes[index] = xmin\n',
            '                        index = index + 1\n',
            '                        boxes[index] = ymin\n',
            '                        index = index + 1\n',
            '                        boxes[index] = xmax\n',
            '                        index = index + 1\n',
            '                        boxes[index] = ymax\n',
            '                        index = index + 1\n',
            '                class_label = getLabel(name.strip())\n',
            '                X.append(img) #save image and label and boxes as array\n',
            '                Y.append(class_label)\n',
            '                bb.append(boxes)\n',
            '                print(img_name+" "+arr[0]+" "+str(boxes)+" "+str(class_label))\n',
            '\n',
            '    X = np.asarray(X)#convert array to numpy format\n',
            '    Y = np.asarray(Y)\n',
            '    bb = np.asarray(bb)\n',
            '    np.save(\'model/X1.npy\', X)#save all processed images\n',
            '    np.save(\'model/Y1.npy\', Y)\n',
            '    np.save(\'model/Z1.npy\', bb)\n',
            '    data = X\n',
            '    labels = Y\n',
            '    bboxes = bb\n',
            '\n',
            'print("Dataset images loaded")\n',
            'print("Total images found in dataset : "+str(data.shape[0]))\n',
            'print("Labels found in dataset : "+str(labels.shape[0]))\n',
        ]
        changed = True
    elif source.startswith('#now load yolov8 from ultralytics packages'):
        cell['source'] = [
            '#now load yolov8 from ultralytics packages\n',
            'from ultralytics import YOLO\n',
            "yolov8_keras_model = load_model('model/v8_model.hdf5', compile=False)\n",
            'yolov8_preds = yolov8_keras_model.predict(testImages)\n',
            'if isinstance(yolov8_preds, list) and len(yolov8_preds) > 1:\n',
            '    yolov8_preds = yolov8_preds[1]\n',
            'predict = np.argmax(yolov8_preds, axis=1)\n',
            'test = np.argmax(testLabels, axis=1)\n',
            'calculateMetrics("YoloV8", predict, test)\n',
            '\n',
            '# load the YOLOv8 PyTorch model for demonstration and image inference\n',
            'yolov8_model = YOLO("model/best.pt")\n',
        ]
        changed = True
    elif source.startswith('#function to predict damage road using extension Yolov8'):
        cell['source'] = [
            '#function to predict damage road using extension Yolov8\n',
            'def damageDetection(yolov8_model, testImage):\n',
            '    frame = cv2.imread(testImage)#read test image\n',
            '    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n',
            '    detections = yolov8_model(frame)[0]#now input test image to extension yolo8 to detect damage road\n',
            '    flag = False\n',
            '    for data in detections.boxes.data.tolist():#now get all damage road detection from predicted output\n',
            '        confidence = data[4]\n',
            '        cls_id = data[5]\n',
            '        if float(confidence) >= 0.3:#if confidence > 0.3 then damage road detected else repaired detected\n',
            '            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])\n',
            '            cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), (0, 255, 0), 2)#put bounding box\n',
            '            cv2.putText(frame, "Road Damaged", ((xmin),(ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)\n',
            '            flag = True\n',
            '        else:\n',
            '            flag = True\n',
            '            cv2.putText(frame, "Road Repaired", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)\n',
            '    if flag == False:\n',
            '        cv2.putText(frame, "Road Repaired", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)\n',
            '    plt.imshow(frame)\n',
            '    plt.show()\n',
        ]
        changed = True

if changed:
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    print('Notebook updated')
else:
    print('No changes applied')
