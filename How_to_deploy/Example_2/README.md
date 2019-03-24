### High level plan as given below:

1.	Export the models-JobLib, Pickl
2.	A python script to load the models and classify [ JobLib, Pickle to load the exported models]
3.	A python web app to serve request and responses [ Flask]
4.	WSGI container to serve the app
5.	Supervisor to get the server process up and running

==================================================================================

Step 1 : Export the model and vectorizer

Step 2 : Create a script to load the models

Step 3 : Add a python function to perform the classification given an input text

Step 4 : Introduce Flask to serve classification through an API end point

Step 5 : Use WSGI to serve the Flask app. 

Step 6 : Create a supervisor config at /etc/supervisor/config.d/your_config.conf

Step 7 : Execute the following commands

	 ```
	 sudo supervisorctl reread
	 sudo supervisorctl update
	 sudo supervisorctl start your_program_name
	 ```

Step 8 : Access it using htttp://ip:port/classifyText 

	 ```
	 Request Body : {"input_text" : "RCB has lost the match by 100 runs"}
	 Response : {"classified_output" : "Sport"}
	 ```

**Note : This has not been tested yet**
