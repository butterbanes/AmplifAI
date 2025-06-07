from ultralytics import YOLO

"""
This class was built as a blueprint to train an AI model on a given .pt file.
At first when I was training the model for the AmplifAI project, I used the yolo10n.pt training model.
At some point I started feeding in the previous iterations of the model training output, albeit only once.
This class is never initialized in the main project program it was written for as it was made to be ran 
isolated from anything else.

@author Ethan Smith
@version 11.01.2024
"""
class GuitarTutorTrainer:

    #with the constructor, we will init the model and train it
    def __init__(self):

        #load .pt model file
        self.model = YOLO('best.pt')

        self.train_results = self.model.train(data='data.yaml', epochs=100, seed=50, patience=150) #adding epochs and adding patience threshold

        #save trained model to a local file for deployment later on
        self.model.export(format="onnx")

#debug statements
if __name__ == "__main__":
    gT = GuitarTutorTrainer()
    print(gT.train_results)
    #print(gT.val_results)
    #print(gT.pred_results)