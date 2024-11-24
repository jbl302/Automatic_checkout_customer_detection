import numpy as np
import os
import glob
import re
import math
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score

class Object:
    def __init__(self):
        self.type = 0
        self.xtop = 0
        self.ytop = 0
        self.xbot = 0
        self.ybot = 0
        self.xres = 0
        self.yres = 0
        
class Image:
    def __init__(self):
        self.number = ''
        self.location = ''
        self.objectList = []
        self.personList = []
        self.labelPath = ''
        self.imgPath = ''
        self.labelList = []
    
class Sequence:
    def __init__(self):
        self.imageDataList = []
        self.label = []
        self.dirName = ''

class Person:
    def __init__(self):
        self.name = ""
        self.xtop = 0
        self.ytop = 0
        self.xbot = 0
        self.ybot = 0
        self.jointLocations = []
        # [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
        # i have taken only 18 locations according to the parse_pickle script.
        #access via 2) index location
        self.pickedUpItems = []
        self.Wallet = 0
        self.inBagItems = []

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def calculateDistance(obj1,obj2):
    if obj1[0] == -1 or obj1[1] == -1 or obj2[0] == -1 or obj2[1] == -1:
        return -1   #return a large distance.
    return math.sqrt( pow(obj1[0]-obj2[0],2) + pow(obj1[1]-obj2[1],2))
    
def calculateCentroid(obj):
    lis = []
    lis.append((float(obj.xtop) + float(obj.xbot))/2)
    lis.append((float(obj.ytop) + float(obj.ybot))/2)
    return lis

def get_bag_and_other_objects(objects):
    bags = []
    others = []
    for item in objects:
        obj = Object()
        if type(item) != type(obj):
            print("error")
            print(type(item))
            print(type(obj))
        if item.type == 8:
            bags.append(item)
        else:
            others.append(item)
    return bags,others
    
def get_closest_bag(objects,person):
    bags, others = get_bag_and_other_objects(objects)
    min_dist = 999999999
    closest_bag = ''
    for item in bags:
        ans = calculateDistance(calculateCentroid(item), calculateCentroid(person))
        if ans < min_dist:
            min_dist = ans
            closest_bag = item
    if closest_bag == '':
        return []
    else:
        others.append(closest_bag)
    return others

def fill_in_missing_objects(modobjects):
    seenobjects= {}
    for obj in modobjects:
        if (obj.type - 1)  not in seenobjects.keys():
            seenobjects[obj.type - 1] = 1
    for i in range(8):
        if i  not in seenobjects.keys():
            nullobj = Object()
            nullobj.type= i + 1
            modobjects.append(nullobj)
            seenobjects[i] = 1
    return modobjects


def generate_person_object_locations_prevframes(sequence):
    person_set = {}
    # person_set is a dictionary mapping person name to the list of tuples (object, person) over the past 5 frames.
    # person_set["karthik"] = [(objects,person) , (objects,person) ,(objects,person) , (objects,person), (objects,person)]
    #there might be missing objects but the person is in all the frames with deafult values added in.
    idx = 0
    if len(sequence.imageDataList) != 5:
        print(" Maintain the length of sequence as 5")
    for item in sequence.imageDataList:
        idx += 1
        objects = item.objectList
        persons = item.personList
        for person in persons:
            objectsmod = get_closest_bag(objects,person)
            objectsmod = fill_in_missing_objects(objectsmod)
            if person.name in person_set.keys():
                lis = person_set[person.name]
                last = lis[-1]
                previdx = last[0]
                while previdx != idx-1:
                    per = Person()
                    lis.append((previdx + 1,[],per))
                    previdx += 1
                lis.append((idx,objectsmod,person))
                person_set[person.name] = lis
            else:
                lis = []
                previdx = 0
                while previdx != idx-1:
                    per = Person()
                    lis.append((previdx + 1,[],per))
                    previdx += 1
                lis.append((idx,objectsmod,person))
                person_set[person.name] = lis
    for item in person_set.keys():
        lis = person_set[item]
        newlis = []
        for it in lis:
            newlis.append( ( it[1] , it[2] ) )
        person_set[item] = newlis
        
    #also filters with only the bag closest to the person added to the objects corresponding to that person.
    # now for each person you have a the set of object locations and pose locations over the past 5 frames.
    return person_set

def generate_featuremap(sequence):
    #output: generates a feature map for each person in the fashion [(person.name, featuremap) , (person.name, featuremap) .. ]
            # feature are generated for the last 5 frames
    person_set = generate_person_object_locations_prevframes(sequence)
    feature_list = []
    
    for item in person_set.keys():
        name = item
        item = person_set[item]
        combined_feature = []
        for frameitem in item:  #iterating over frames and appending features
            feature1 = []
            feature2 = []
            feature3 = []
            feature4 = []
            joint_locations = frameitem[1].jointLocations
            if len(joint_locations) == 0:
                shoulder_pos_r = (0 , 0 )
                wrist_pos_r = (0 , 0 )
                shoulder_pos_l = (0 , 0 )
                wrist_pos_l = (0 , 0 )
                head_loc = (0 , 0 )
            else:
                shoulder_pos_r = ( joint_locations[4] , joint_locations[5] )
                wrist_pos_r = ( joint_locations[8] , joint_locations[9] )
                shoulder_pos_l = ( joint_locations[10] , joint_locations[11] )
                wrist_pos_l = ( joint_locations[14] , joint_locations[15] )
                head_loc = ( joint_locations[0] , joint_locations[1] )
            #feature_1 shoulder wrist distance
            feature1.append(calculateDistance(wrist_pos_l, shoulder_pos_l))
            feature1.append(calculateDistance(wrist_pos_r, shoulder_pos_r))
    
            #feature_2 
            count = 7 #maximum number of objects in a frame
            bags, others = get_bag_and_other_objects(frameitem[0])
            for obj in others:
                count = count - 1
                feature2.append(calculateDistance(wrist_pos_l,calculateCentroid(obj)))
                feature2.append(calculateDistance(wrist_pos_r,calculateCentroid(obj)))
                feature3.append(calculateDistance(head_loc,calculateCentroid(obj)))
                feature4.append(calculateDistance(calculateCentroid(bags[0]),calculateCentroid(obj)))
            while count > 0:
                feature2.append(0)
                feature2.append(0)
                feature3.append(0)
                feature4.append(0)
                count = count - 1
            feature = feature1 + feature2 + feature3 + feature4
            combined_feature.extend(feature)
        feature_list.append((name,combined_feature))     
    return feature_list
         
def generate_data(all_sequences):
    all_features = []
    all_labels = []
    for seq in all_sequences:
        features = generate_featuremap(seq)
        #print(features)
        labels = seq.label
        #print(labels)
        if(len(labels) == len(features)):
            for it in range(0,len(labels)):
                #print(len(features[it][1]))
                if len(features[it][1]) == 150: #calculated from the constant value. Make sure this is 150 else you have more objects or blank persons added.
                    all_features.append(features[it][1])
                    all_labels.append(labels[it])
    return all_features,all_labels

def discretize(labels):
    newlabels=[]
    for item in labels:
        print(item)
        if item == "picking":
            newlabels.append(0)
        elif item == "placing":
            newlabels.append(1)
        elif item == "cart":
            newlabels.append(2)
        elif item == "idle":
            newlabels.append(3)
        else:
            print("ERROR when discretizing.")
    return newlabels

#saving the SVM's model

from joblib import dump
def linear_classification(features,labels):
#    try:
    data = features
    labels = np.asarray(discretize(labels))
    kf = KFold(n_splits=4,shuffle=True)
    conf_mat=np.full((4,4),0)
    scores=[]
    for train_index, test_index in kf.split(data):
        train_d = []
        train_l = []
        test_d = []
        test_l = []
        for it in range(0,len(features)):
            if it in train_index:
                train_d.append(features[it])
                train_l.append(labels[it])
            if it in test_index:
                test_d.append(features[it])
                test_l.append(labels[it])
        print("done")
        train_d = np.asarray(train_d)
        #print(train_d.shape)
        test_d = np.asarray(test_d)
        #print(test_d.shape)
        global clf
        clf = svm.LinearSVC(C=3,max_iter = 10000)
        #clf = svm.SVC(gamma='scale', decision_function_shape='ovo',C = 3.5 , max_iter = 10000,degree = 5)
        dump(clf, 'activity_recognition.joblib') 
        clf.fit(train_d, train_l)

        ftrain_pred=clf.predict(train_d)
        ftest_pred=clf.predict(test_d)
        cur_score=accuracy_score(test_l,ftest_pred, normalize=True)
        scores.append(cur_score)
        train_score=accuracy_score(train_l,ftrain_pred,normalize=True)
        print(train_score,cur_score)
        #for j in range(len(test_l)):
        #    if(ftest_pred[j]!=test_l[j]):
        #        print("misclassified - ",ftest[j,0]," where ",num_to_exer(ftest[j,1])," as ",num_to_exer(ftest_pred[j]))
        cur_matrix=confusion_matrix(test_l,ftest_pred)
        cur_matrix=np.asarray(cur_matrix)
        conf_mat=np.add(conf_mat, cur_matrix)
    scores=np.asarray(scores)    
    print(scores.mean())
    print(conf_mat)           
#    except:
#        print("error")     



def loadDir():
    all_sequences = []
    imagedir = './mod-data/'
    labeldir = './labels'
    persondir = './persondata'
    dirList = sorted_alphanumeric(glob.glob(os.path.join(imagedir, '**')))
    if len(dirList) == 0:
        print("No directories found")
        
    for directory in dirList:
        cursequence = Sequence()
        cursequence.dirName = directory
        try:
            targetlabelfile = directory + "/label.txt"
            targetlabel = open(targetlabelfile, 'r')
        except IOError:
            print("Error: can\'t find file or read data from label file", directory)
        tar_labels=[]
        for eachline in targetlabel:
            eachline = eachline.rstrip('\n')
            tar_labels.append(eachline)
        cursequence.label = tar_labels
        
        imageList = sorted_alphanumeric(glob.glob(os.path.join(directory, '*.jpg')))
        num = 0
        imgdatalist = []
        for image in imageList:
            curimage = Image()
            
            curimage.persons = len(tar_labels)  #REMOVE THIS ONCE YOU GET JOINT LOCATIONS
            curimage.path = image
            curimage.number = num
            num = num + 1  
            segments = image.split('/')
            imgname = segments[len(segments) - 1]
            imgsegments = imgname.split('.')
            imgnumber = imgsegments[0]
            labelname = imgnumber + ".txt"
            
            
            try:
                location = labeldir +'/'+ labelname
                curimage.labelPath = location
                labelFile = open(location , 'r')
                objects = []
                for line in labelFile:
                    curobject = Object()
                    input_numbers = line.split(' ')
                    if len(input_numbers) == 7:
                        curobject.type = int(input_numbers[0])
                        curobject.xtop = int(input_numbers[1])
                        curobject.ytop = int(input_numbers[2])
                        curobject.xbot = int(input_numbers[3])
                        curobject.ybot = int(input_numbers[4])
                        curobject.xres = int(input_numbers[5])
                        curobject.yres = int(input_numbers[6])
                    else:
                        print("Read ERROR len not 7 " , labelname)
                    objects.append(curobject)
                curimage.objectList = objects
            except IOError:
               print("Error: can\'t find file or read data")
            
            
            persons = []
            #write code to get the joint locations
            try:
                location = persondir +'/'+ labelname
                personFile = open(location , 'r')
                for line in personFile:
                    curperson= Person()
                    input_numbers = line.split(' ')
                    #print( input_numbers)
                    if len(input_numbers) == 42:   #as it inlcudes '\n' at the end. usually its 41
                        curperson.name = input_numbers[0]
                        curperson.xtop = int(input_numbers[1])
                        curperson.ytop = int(input_numbers[2])
                        curperson.xbot = int(input_numbers[3])
                        curperson.ybot = int(input_numbers[4])
                        for idx in range(5,41):
                            curperson.jointLocations.append(int(input_numbers[idx]))
                    else:
                        print("Read ERROR len not 41 " , labelname)
                    persons.append(curperson)
                curimage.personList = persons
            except IOError:
               print("Error: can\'t find file or read data " , labelname)
            
            imgdatalist.append(curimage)
        cursequence.imageDataList = imgdatalist    
        all_sequences.append(cursequence)
    return all_sequences
        
all_sequences = loadDir()   
all_features, all_labels = generate_data(all_sequences)

from sklearn.preprocessing import MinMaxScaler  
scaler = MinMaxScaler(feature_range = (0, 1))
all_features_scaled = scaler.fit_transform(all_features)  


linear_classification(all_features,all_labels)
