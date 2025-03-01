#function to get the images.
def loading_image(path):
    path = [os.path.join(path, file) for file in os.listdir(path)]
    list_images = []
    for path_file in path:
        photo = Image.open(path_file)
        list_images.append(photo.convert('RGB'))
    return list_images

#function loading five images per classes 
def loading_5_per_class_lab(path, label):
    liste_image = []
    dict_count_lab = {}
    label_img = []
    for i in range(len(label)):
        if label[i] not in dict_count_lab:
            dict_count_lab[label[i]] = 0    
    for file in os.listdir(path):
        ident = file[0:4]
        if dict_count_lab [ident] == 5:
            continue
        else:
            path_file = os.path.join(path, file)
            photo = Image.open(path_file)
            liste_image.append(photo.convert('RGB'))
            dict_count_lab[ident] += 1
            label_img.append(ident)
    return liste_image, label_img

#function to get the labels
def load_label(path):
  label = []
  for file in os.listdir(path):
    lab = file[0:4] 
    label.append(lab)
  return label

#generate the dico used for the triplet.
def generate_dico(image, label):
  dico = {}
  for i in range(len(image)):
    img = image[i]
    lab = label[i]
    if lab not in dico:
      dico[lab] = []
    dico[lab].append(image[i])
  return dico

#function creating the triplet anchor, positive and   
def create_triplet(image,dico):
  list_anchor = []
  list_pos = []
  list_neg = []
  for lab in dico: 
    list_idx_pos = [k for k in range(len(dico[lab]))]
    label_neg = [labe for labe in dico if labe != lab]
    label_neg_idx = np.random.choice([k for k in range(len(label_neg))])
    list_idx_neg = [k for k in range(len(dico[label_neg[label_neg_idx]]))]
    anch_idx = np.random.choice(list_idx_pos)
    pos_indx = np.random.choice([k for k in list_idx_pos if k!= anch_idx])
    neg_idx = np.random.choice(list_idx_neg)
    anchor = dico[lab][anch_idx]
    posit = dico[lab][pos_indx]
    negat = dico[label_neg[label_neg_idx]][neg_idx]
    list_anchor.append(anchor)
    list_pos.append(posit)
    list_neg.append(negat)
  return list_anchor, list_pos, list_neg

# normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize 
    transforms.ToTensor(),           # Convert PIL Image to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])
