import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import make_grid
import torch
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
DATA_MEAN = (0.4914, 0.4822, 0.4465)
DATA_STD = (0.247, 0.2435, 0.2616)


class Plots:
    def __init__(self):
        pass

    def sampleVisual(dataset):
        batch = next(iter(dataset))
        images, labels = batch
        images = images["image"]
        batch_grid = make_grid(images)
        images = batch_grid.numpy().transpose((1, 2, 0))  # (C, H, W) --> (H, W, C)
        # Convert mean and std to numpy array
        mean = np.asarray(DATA_MEAN)
        std = np.asarray(DATA_STD)
        # unnormalize the image
        images = DATA_STD * images + DATA_MEAN
        images = np.clip(images, 0, 1)
        fig = plt.figure()  # Create a new figure
        fig.set_figheight(15)
        fig.set_figwidth(15)
        ax = fig.add_subplot(111)
        ax.axis("off")  # Sqitch off the axis
        ax.imshow(images)
        #return plt.imshow(batch_grid[0].squeeze(), cmap='gray_r')
        # return ax.imshow(images)

    def plotting(model, loader, device):
        wrong_images = []
        wrong_label = []
        correct_label = []

        with torch.no_grad():
            for img, label in loader:
                img, label = img.to(device), label.to(device)
                pred_label = model(img.to(device))
                pred = pred_label.argmax(dim=1, keepdim=True)

                wrong_pred = (pred.eq(target.view_as(pred)) == False)
                wrong_images.append(data[wrong_pred])
                wrong_label.append(pred[wrong_pred])
                correct_label.append(target.view_as(pred)[wrong_pred])

                wrong_predictions = list(
                    zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
                fig = plt.figure(figsize=(8, 10))
                fig.tight_layout()
                for i, (img, pred, correct) in enumerate(wrong_predictions[:10]):
                    img, pred, target = img.cpu().numpy(), pred.cpu(), correct.cpu()
                    ax = fig.add_subplot(5, 2, i+1)
                    ax.axis('off')
                    ax.set_title(
                        f'\nactual {target.item()}\npredicted {pred.item()}', fontsize=10)
                    ax.imshow(img.squeeze(), cmap='gray_r')

                plt.show()
            return len(wrong_predictions)

    def stat_graph(train_acc, train_losses, test_acc, test_losses):
      fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
      ax1 = ax[0, 0]
      ax1.set_title("TrainAccuracy")
      ax1.plot(train_acc, color="blue")
      ax2 = ax[0, 1]
      ax2.set_title("TestAccuracy")
      ax2.plot(test_acc, color="blue")
      ax3 = ax[1, 0]
      ax3.set_title("TrainLoss")
      ax3.plot(train_losses, color="blue")
      ax4 = ax[1, 1]
      ax4.set_title("TestLoss")
      ax4.plot(test_losses, color="blue")
      plt.show()

    def miscImages(model, test_loader, device):
      classes = ('plane', 'car', 'bird', 'cat',
                 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
      wrong_images = []
      wrong_label = []
      correct_label = []
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data["image"].to(device), target.to(device)
              output = model(data)
              _, predicted = torch.max(output.data, 1)
              wrong_pred = (predicted.eq(target.view_as(predicted)) == False)
              wrong_images.append(data[wrong_pred])
              wrong_label.append(predicted[wrong_pred])
              correct_label.append(target.view_as(predicted)[wrong_pred])

              wrong_predictions = list(
                  zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
          print(f'Total wrong predictions are {len(wrong_predictions)}')

          fig = plt.figure(figsize=(15, 10))
          fig.tight_layout()
          for i, (img, pred, correct) in enumerate(wrong_predictions[:50]):
            img, pred, target = img.cpu().numpy(), pred.cpu(), correct
            ax = fig.add_subplot(5, 10, i+1)
            ax.axis('off')
            ax.set_title(
                f'\nactual {classes[target.item()]}\npredicted {classes[pred.item()]}', fontsize=10)
            ax.imshow(img.transpose(1, 2, 0).squeeze().astype(np.uint8),
                      cmap='gray_r', vmin=0, vmax=255)
          plt.show()

    def miscImages_1(model,  test_loader, device):
      model.eval()
      test_loss = 0
      incorrect = 0
      img_grid = []
      fig = plt.figure(figsize=(8, 10))
      fig.tight_layout()
      with torch.no_grad():
          count = 10
          for data, target in test_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              # get the index of the max log-probability
              pred = output.argmax(dim=1, keepdim=True)

              for i in range(len(target)):
                if pred[i].item() != target[i]:
                  incorrect += 1
                  print('\n\n{} [ Predicted Value: {}, Actual Value: {} ]'.format(
                      incorrect, pred[i].item(), target[i], ))
                  ax = fig.add_subplot(5, 2, i)
                  ax.axis('off')
                  ax.imshow(data[i].cpu().numpy().transpose(
                      1, 2, 0).squeeze(), cmap='gray_r')
                  plt.show()
                  count = count - 1
                if count == 0:
                  break
              if count == 0:
                break


    def plot_grad_cam(cam,images,target_category,denorm):
    

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=images, target_category=target_category,aug_smooth=True,eigen_smooth=False)
        
        plot_images = torch.clone(images).detach() # Create Copy of Input Images
        denorm_image_list = list(map(denorm,plot_images)) # Denormalise Images
        denorm_tensor = torch.stack(denorm_image_list,dim=0) # Create Batched Tensor
        
        # Change order for Plotting
        rgb_tensor  = denorm_tensor.permute(0, 2, 3, 1).cpu().numpy() 
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        

        num_images = images.shape[0]
        fig = plt.figure(figsize=(8, 8))
        # fig.tight_layout()
        layout_id =1 
        
        
        for idx,(img,img_cam) in enumerate(zip(images,grayscale_cam)) : 
            
            visualization = show_cam_on_image(rgb_tensor[idx], img_cam) #Pass Denorm Image for SuperImposing
            
            # Normal Images Plot
            ax = fig.add_subplot(num_images, 2, layout_id)
            # ax.axis('off')
            
            ax.set_title("Input Image")
            ax.imshow(img.astype('uint8'),vmin=0, vmax=255)
            layout_id+=1

            # Cam Output Plot
            ax = fig.add_subplot(num_images, 2, layout_id)
            # ax.axis('off')
            ax.set_title("Cam Image")
            ax.imshow(visualization.astype('uint8'),vmin=0, vmax=255)
            layout_id+=1
        
