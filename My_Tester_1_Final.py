
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
# from CGNet_Model_MyDecoder__SA import *
# from ChangeFormerV4_MyEncoder import *
from CGNet_Model import *
# from SiamUnet_diff import *
# from SiamUnet_EF import *


from My_Trainer_1_Final2 import test_loader
import time

# Set CUDA_LAUNCH_BLOCKING environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import random

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


############## Test the model ##################
############## Test the model ##################

############## Confusion Matrix ##################

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


############## Test the model ##################
from torchvision.utils import save_image

if __name__ == "__main__":

    seeding(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"


    print(device)

    model = CGNet()
    model=model.to(device)
    model.load_state_dict(torch.load("E://VS Codes Testing things_2//test_6-3-2025_CGNet - 2_w_CEFF//checkpoints_SYSU//checkpoint1-50epochs//ResUnet13.pth"))
    model.eval()
    test_results_path="E://VS Codes Testing things_2//test_6-3-2025_CGNet - 2_w_CEFF//test results_SYSU//test_results1-50epochs"
    os.makedirs(test_results_path,exist_ok=True)
    TP=0
    TN=0
    FP=0
    FN=0
    start_time = time.time()
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
            pre_tensor = pre_tensor.to(device)
            post_tensor = post_tensor.to(device)
            label_tensor = label_tensor.to(device)
            probs = model(pre_tensor, post_tensor)
            prediction = torch.where(probs> 0.3604056239128113, 1.0,0.0)
            true_positives, false_positives, true_negatives, false_negatives = confusion(prediction,label_tensor)
            TP+=true_positives
            TN+=true_negatives
            FP+=false_positives
            FN+=false_negatives
            for i in range(prediction.shape[0]):
                save_image(prediction[i,:,:,:].cpu(), os.path.join(test_results_path, fname[i]))
        end_time = time.time()
        inference_time = (end_time - start_time)/len(test_loader)
    ################## Visualize the results ##################
    import matplotlib.pyplot as plt
    import numpy as np

    pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
    fig=plt.figure(figsize=(10,10))

    # output_image = np.zeros((4, 1, 256, 256), dtype=np.uint8)
    # output_image0 = output_image[0]
    # print ("output_image0:   ", output_image0.shape)
    # # prediction =prediction.numpy()#[0,:,:,:].numpy()#permute(1,2,0).numpy()
    # print ("prediction:  ", prediction.shape)
    # # label_tensor0 = label_tensor[0,:,:,:].numpy()#permute(1,2,0).numpy()
    # # # output_image = prediction[0,:,:,:]
    # output_image0[(label_tensor0 == 1) & (prediction0 == 1)] = 255#[255, 255, 255]  # White
    # output_image0[(label_tensor0 == 0) & (prediction0 == 0)] = [0, 0, 0]        # Black
    # output_image0[(label_tensor0 == 0) & (prediction0 == 1)] = [255, 0, 0]      # Red
    # output_image0[(label_tensor0 == 1) & (prediction0 == 0)] = [0, 0, 255]      # Blue



    preplot=fig.add_subplot(441)
    preplot.imshow(pre_tensor[0,:,:,:].permute(1,2,0).numpy())
    preplot.set_title("Pre-change")

    postplot=fig.add_subplot(442)
    postplot.set_title("Post-change")
    postplot.imshow(post_tensor[0,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(443)
    postplot.set_title("Label-tensor")
    postplot.imshow(label_tensor[0,:,:,:].permute(1,2,0).numpy(), cmap='gray') 

    labelplot=fig.add_subplot(444)
    labelplot.set_title("Prediction")
    # labelplot.imshow(prediction[0,:,:,:].permute(1,2,0).cpu().numpy(), cmap='gray') 

    ground_truth0 = label_tensor[0, :, :, :].numpy()  # Assuming label_tensor is the ground truth
    prediction0 = prediction[0, :, :, :].cpu().numpy()  # Assuming prediction is defined and matches the shape
    ground_truth0 = ground_truth0.squeeze()  # Assuming label_tensor is the ground truth
    prediction0 = prediction0.squeeze()  # Assuming prediction is defined and matches the shape
    # Create an RGB image for visualization
    output_image = np.zeros((*ground_truth0.shape, 3), dtype=np.uint8) 
    output_image[(ground_truth0 == 1) & (prediction0 == 1)] = [255, 255, 255]       # White: True Positives  
    output_image[(ground_truth0 == 0) & (prediction0 == 0)] = [0, 0, 0]             # Black: True Negatives
    output_image[(ground_truth0 == 0) & (prediction0 == 1)] = [255, 0, 0]           # Red: False Positives
    output_image[(ground_truth0 == 1) & (prediction0 == 0)] = [0, 0, 255]           # Blue: False Negatives
    labelplot.imshow(output_image)





    # transforms.ToPILImage()(pre_tensor[0,:,:,:])
    # transforms.ToPILImage()(post_tensor[0,:,:,:])
    # transforms.ToPILImage()(label_tensor[0,:,:,:])
    print(f'fname={fname[0]}')


    preplot=fig.add_subplot(445)
    preplot.imshow(pre_tensor[1,:,:,:].permute(1,2,0).numpy())
    # preplot.set_title("pre-change")

    postplot=fig.add_subplot(446)
    # postplot.set_title("post-change")
    postplot.imshow(post_tensor[1,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(447)
    # postplot.set_title("label-tensor")
    postplot.imshow(label_tensor[1,:,:,:].permute(1,2,0).numpy(), cmap='gray') 

    labelplot=fig.add_subplot(448)
    # labelplot.set_title("prediction")
    # labelplot.imshow(prediction[1,:,:,:].permute(1,2,0).cpu().numpy(), cmap='gray') 
    print(f'fname={fname[1]}')
    
    ground_truth1 = label_tensor[1, :, :, :].numpy()  # Assuming label_tensor is the ground truth
    prediction1 = prediction[1, :, :, :].cpu().numpy()  # Assuming prediction is defined and matches the shape
    ground_truth1 = ground_truth1.squeeze()  # Assuming label_tensor is the ground truth
    prediction1 = prediction1.squeeze()  # Assuming prediction is defined and matches the shape
    # Create an RGB image for visualization
    output_image = np.zeros((*ground_truth1.shape, 3), dtype=np.uint8) 
    output_image[(ground_truth1 == 1) & (prediction1 == 1)] = [255, 255, 255]       # White: True Positives  
    output_image[(ground_truth1 == 0) & (prediction1 == 0)] = [0, 0, 0]             # Black: True Negatives
    output_image[(ground_truth1 == 0) & (prediction1 == 1)] = [255, 0, 0]           # Red: False Positives
    output_image[(ground_truth1 == 1) & (prediction1 == 0)] = [0, 0, 255]           # Blue: False Negatives
    labelplot.imshow(output_image)





    preplot=fig.add_subplot(449)
    preplot.imshow(pre_tensor[3,:,:,:].permute(1,2,0).numpy())
    # preplot.set_title("pre-change")

    postplot=fig.add_subplot(4,4,10)
    # postplot.set_title("post-change")
    postplot.imshow(post_tensor[3,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(4,4,11)
    # postplot.set_title("label-tensor")
    postplot.imshow(label_tensor[3,:,:,:].permute(1,2,0).numpy(), cmap='gray') 

    labelplot=fig.add_subplot(4,4,12)
    # labelplot.set_title("prediction")
    # labelplot.imshow(prediction[3,:,:,:].permute(1,2,0).cpu().numpy(),cmap='gray') 
    print(f'fname={fname[3]}')

    ground_truth3 = label_tensor[3, :, :, :].numpy()  # Assuming label_tensor is the ground truth
    prediction3 = prediction[3, :, :, :].cpu().numpy()  # Assuming prediction is defined and matches the shape
    ground_truth3 = ground_truth3.squeeze()  # Assuming label_tensor is the ground truth
    prediction3 = prediction3.squeeze()  # Assuming prediction is defined and matches the shape
    # Create an RGB image for visualization
    output_image = np.zeros((*ground_truth3.shape, 3), dtype=np.uint8) 
    output_image[(ground_truth3 == 1) & (prediction3 == 1)] = [255, 255, 255]       # White: True Positives  
    output_image[(ground_truth3 == 0) & (prediction3 == 0)] = [0, 0, 0]             # Black: True Negatives
    output_image[(ground_truth3 == 0) & (prediction3 == 1)] = [255, 0, 0]           # Red: False Positives
    output_image[(ground_truth3 == 1) & (prediction3 == 0)] = [0, 0, 255]           # Blue: False Negatives
    labelplot.imshow(output_image)



        
    preplot=fig.add_subplot(4,4,13)
    preplot.imshow(pre_tensor[2,:,:,:].permute(1,2,0).numpy())
    # preplot.set_title("pre-change")

    postplot=fig.add_subplot(4,4,14)
    # postplot.set_title("post-change")
    postplot.imshow(post_tensor[2,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(4,4,15)
    # postplot.set_title("label-tensor")
    postplot.imshow(label_tensor[2,:,:,:].permute(1,2,0).numpy(), cmap='gray') 

    labelplot=fig.add_subplot(4,4,16)
    # labelplot.set_title("prediction")
    # labelplot.imshow(prediction[2,:,:,:].permute(1,2,0).cpu().numpy(),cmap='gray') 
    print(f'fname={fname[2]}')

    ground_truth2 = label_tensor[2, :, :, :].numpy()  # Assuming label_tensor is the ground truth
    prediction2 = prediction[2, :, :, :].cpu().numpy()  # Assuming prediction is defined and matches the shape
    ground_truth2 = ground_truth2.squeeze()  # Assuming label_tensor is the ground truth
    prediction2 = prediction2.squeeze()  # Assuming prediction is defined and matches the shape
    # Create an RGB image for visualization
    output_image = np.zeros((*ground_truth2.shape, 3), dtype=np.uint8) 
    output_image[(ground_truth2 == 1) & (prediction2 == 1)] = [255, 255, 255]       # White: True Positives  
    output_image[(ground_truth2 == 0) & (prediction2 == 0)] = [0, 0, 0]             # Black: True Negatives
    output_image[(ground_truth2 == 0) & (prediction2 == 1)] = [255, 0, 0]           # Red: False Positives
    output_image[(ground_truth2 == 1) & (prediction2 == 0)] = [0, 0, 255]           # Blue: False Negatives
    labelplot.imshow(output_image)

    plt.show()



    pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
    fig=plt.figure(figsize=(10,10))

    preplot=fig.add_subplot(441)
    preplot.imshow(pre_tensor[0,:,:,:].permute(1,2,0).numpy())
    preplot.set_title("Pre-change")

    postplot=fig.add_subplot(442)
    postplot.set_title("Post-change")
    postplot.imshow(post_tensor[0,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(443)
    postplot.set_title("Label-tensor")
    postplot.imshow(label_tensor[0,:,:,:].permute(1,2,0).numpy(), cmap='gray') 

    labelplot=fig.add_subplot(444)
    labelplot.set_title("Prediction")
    labelplot.imshow(prediction[0,:,:,:].permute(1,2,0).cpu().numpy(), cmap='gray') 

    # transforms.ToPILImage()(pre_tensor[0,:,:,:])
    # transforms.ToPILImage()(post_tensor[0,:,:,:])
    # transforms.ToPILImage()(label_tensor[0,:,:,:])
    print(f'fname={fname[0]}')


    preplot=fig.add_subplot(445)
    preplot.imshow(pre_tensor[1,:,:,:].permute(1,2,0).numpy())
    # preplot.set_title("pre-change")

    postplot=fig.add_subplot(446)
    # postplot.set_title("post-change")
    postplot.imshow(post_tensor[1,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(447)
    # postplot.set_title("label-tensor")
    postplot.imshow(label_tensor[1,:,:,:].permute(1,2,0).numpy(), cmap='gray') 

    labelplot=fig.add_subplot(448)
    # labelplot.set_title("prediction")
    labelplot.imshow(prediction[1,:,:,:].permute(1,2,0).cpu().numpy(), cmap='gray') 
    print(f'fname={fname[1]}')


    preplot=fig.add_subplot(449)
    preplot.imshow(pre_tensor[3,:,:,:].permute(1,2,0).numpy())
    # preplot.set_title("pre-change")

    postplot=fig.add_subplot(4,4,10)
    # postplot.set_title("post-change")
    postplot.imshow(post_tensor[3,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(4,4,11)
    # postplot.set_title("label-tensor")
    postplot.imshow(label_tensor[3,:,:,:].permute(1,2,0).numpy(), cmap='gray') 

    labelplot=fig.add_subplot(4,4,12)
    # labelplot.set_title("prediction")
    labelplot.imshow(prediction[3,:,:,:].permute(1,2,0).cpu().numpy(),cmap='gray') 
    print(f'fname={fname[3]}')

       
    preplot=fig.add_subplot(4,4,13)
    preplot.imshow(pre_tensor[2,:,:,:].permute(1,2,0).numpy())
    # preplot.set_title("pre-change")

    postplot=fig.add_subplot(4,4,14)
    # postplot.set_title("post-change")
    postplot.imshow(post_tensor[2,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(4,4,15)
    # postplot.set_title("label-tensor")
    postplot.imshow(label_tensor[2,:,:,:].permute(1,2,0).numpy(), cmap='gray') 

    labelplot=fig.add_subplot(4,4,16)
    # labelplot.set_title("prediction")
    labelplot.imshow(prediction[2,:,:,:].permute(1,2,0).cpu().numpy(),cmap='gray') 
    print(f'fname={fname[2]}')



    plt.show()

    ############# Calculate Inference time of Model #############
    # dummy_x1 = torch.rand([1,3,256,256]).to(device)
    # dummy_x2 = torch.rand([1,3,256,256]).to(device)
    # model.eval()
    # start_time = time.time()
    # y = model (dummy_x1, dummy_x2)
    # end_time = time.time()
    # inference_time = end_time - start_time
    # print(f'Inference Time: {inference_time:.6f} seconds')

    ############# Calculate the metrics ##############
    print ("TP: ",TP, "TN: ", TN, "FP: ", FP, "FN: ", FN)
    OA=(TP+TN)/(TP+TN+FP+FN)   # Observed Accuracy
    Precision=TP/(TP+FP)
    Recall=TP/(TP+FN)
    F1_score=2*Precision*Recall/(Precision+Recall)
    IoU1 =TP/(TP+FP+FN)
    IoU2 =TN/(TN+FP+FN)
    mIoU = (IoU1+IoU2)/2
    Spec = TN/(TN+FP)

    P1 = TP/(TP+FP)
    R1 = TP/(TP+FN)
    P2 = TN /(TN+FN)
    R2 = TN /(TN+FP)
    F1 = 2*P1*R1/(P1+R1)
    F2 = 2*P2*R2/(P2+R2)
    microF = (F1+F2)/2

    total = TP+TN+FP+FN 
    PRE = ((TP+FP)*(TP+FN) + (TN+FN)*(TN+FP)) / (total ** 2)  # Expected accuracy
    Kappa = (OA - PRE) / (1 - PRE)

    print ("                                                             ")
    print("--------------------------Results-----------------------------")
    print(f'Inference Time: {inference_time:.6f} sec, OA={OA:.4f}, Precision={Precision:.4f}, Recall/Sensitivity={Recall:.4f}, Specificity={Spec:.4f}, Kappa={Kappa:.4f}, mIoU={mIoU:.4f} IoU={IoU1:.4f}, mF1-score={microF:.4f}, F1-score={F1_score:.4f}' )
    print(f'Inference Time, OA, Precision, Recall/Sensitivity, Specificity, Kappa, mIoU, IoU, mF1-score, F1-score' )
    print(f'{inference_time:.4f}, {OA:.4f}, {Precision:.4f}, {Recall:.4f}, {Spec:.4f},{Kappa:.4f}, {mIoU:.4f} , {IoU1:.4f}, {microF:.4f}, {F1_score:.4f}' )
    print ("-------------------------------------------------------------")


    ################## ROC curve ######################
