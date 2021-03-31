
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, send
import json
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth
from NeuralNet import Model
import pymongo
import time

app = Flask(__name__)


CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')


# For firebase auth
cred = credentials.Certificate("./serviceAccountKey.json")
firebase_app = firebase_admin.initialize_app(cred)


class User:
	def __init__(self):
		self.modelObj = Model()
		self.Username = None
		self.clients = 0
		self.training = False

		self.myclient = pymongo.MongoClient("mongodb://localhost:27017/")
		self.mydb = self.myclient["PlaygroundDB"]
		self.mycol = self.mydb['Users']
		


user = User()



def isSecure(id_token):

	decoded_token = auth.verify_id_token(id_token)
	uid = decoded_token['uid']

	user.Username = uid
	return True



@socketio.on('get-dataset')
def send_dataset(message):
	if(isSecure(message['sessionId'])):
		user.modelObj.regen_dataset()
		Y_test = user.modelObj.Y_test
		X_test = user.modelObj.X_test

		Y_test = json.dumps(Y_test.tolist())
		X_test = json.dumps(X_test.tolist())

		emit('outputImage', {'Y_test': Y_test, 'coordinates': X_test})




@socketio.on('new-message')
def handle_message(message):

	if(isSecure(message['sessionId'])):
		if (not user.training or not message['training']):

			user.training = message['training']
			
			if user.training:
				send_data()
			else:
				emit('outputImage', {'training': 'finished training'})


def send_data():
	user.modelObj.create_model(user.mycol.find_one({"userName": user.Username}))

	epoch = 0
	user.modelObj.reset_model()

	while user.training:
		user.modelObj.train_model(epoch)
		epoch += 1
		prediction = user.modelObj.prediction.numpy()
		coords = user.modelObj.X_test

		prediction = json.dumps(prediction.tolist())
		coords = json.dumps(coords.tolist())

		emit('outputImage', {'Y_test': prediction, 'coordinates': coords, 'epoch': epoch})




@app.route("/api/getModel/<sessionId>", methods=["GET"])
def getModel(sessionId):
	if request.method == "GET" and isSecure(sessionId):
		model = user.mycol.find_one({"userName": user.Username})
		model.pop("_id")
		model.pop("userName")
		return jsonify(data= model)

	return jsonify(data="Something went wrong managin layer")




@app.route("/api/manageLayer/<sessionId>", methods=["POST"])
def manageLayer(sessionId):
	

	if request.method == "POST" and isSecure(sessionId):
		x = user.mycol.delete_many({"userName": user.Username})

		myquery = request.form.get("model")

		myquery = json.loads(request.form.get("model"))
		myquery.update({"userName": user.Username})
		x = user.mycol.insert_one(myquery)
		return jsonify(data= "finished managing layer")


	return jsonify(data="Something went wrong managing layer")


if __name__ == "__main__":
	app.run(debug=False)
