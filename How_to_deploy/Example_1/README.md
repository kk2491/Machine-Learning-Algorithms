### Example 1 : Deploying the Trained machine learning algorithm

1. Train the machine learning model.

2. Save the model weights in a pickle file.

3. Create a web server and deploy the trained model by loading the pickle file.

Execution Steps:

1. Run ```model.py```, this will train the model and save the model in a pickle file. <br /> You can change the dataset and model based on your requirement.

2. Run ```deploy.py```, this will start web server and loads the model during the initialization. <br /> If you try to open the webserver URL you may encounter some error. <br /> So once server is started open a new terminal and run the below command to pass the data to webserver.
```curl -X POST http://127.0.0.1:5000/predict --data '{ "feature_array
" : [7.4,0.66,0,1.8,0.075,13,40,0.9978,3.51,0.56,9.4] }' --header "Content-Type: application/json"
``` 
This predicts the output.



