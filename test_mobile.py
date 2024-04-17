# import necessary packages

import cv2
import numpy as np
import mediapipe as mp

import math
import pyglet
from threading import Thread

def sound_alarm(path = 'alarm.wav'):
	# play an alarm sound
	music = pyglet.resource.media('alarm.wav')
	music.play()
	pyglet.app.run()

ALARM_ON = False
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands= 2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=0.1, circle_radius=0.5)

f = mp_face_mesh.FaceMesh(
	static_image_mode=True,
	max_num_faces=1,
	min_detection_confidence=0.5)

alert = ''


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
	# Read each frame from the webcam
	_, frame = cap.read()

	x, y, c = frame.shape

	# Flip the frame vertically
	frame = cv2.flip(frame, 1)
	framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Get hand landmark prediction
	result = hands.process(framergb)
	f_result = f.process(framergb)   # Feed the image to hands module to process and save to 'result'


	if f_result.multi_face_landmarks:

		f_landmarks = []
		for facelms in f_result.multi_face_landmarks:
			for lm in facelms.landmark:
				lmx = int(lm.x * x)
				lmy = int(lm.y * y)
				f_landmarks.append([lmx, lmy])    # Append the values to the empty list created

			mpDraw.draw_landmarks(frame, facelms, mp_face_mesh.FACEMESH_CONTOURS)



	# post process the result
	if result.multi_hand_landmarks:
		landmarks = []
		for handslms in result.multi_hand_landmarks:
			for lm in handslms.landmark:
				# print(id, lm)
				lmx = int(lm.x * x)
				lmy = int(lm.y * y)

				landmarks.append([lmx, lmy])

			# Drawing landmarks on frames
			mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)


			print(len(landmarks))
			print(len(f_landmarks))

		finger_point = landmarks[4]
		ear_point = f_landmarks[234]
		alert = 'normal'
		color = (0,255,0)
		distance = landmarks[4][0] - f_landmarks[234][0]
		distance = math.sqrt(((finger_point[0] - ear_point[0]) ** 2) + ((finger_point[1] - ear_point[1]) ** 2))
		print(distance)
		
		if distance < 20:
			alert = 'do not use phone'
			color = (0,0,255)
			if not ALARM_ON:
				ALARM_ON = True
				t = Thread(target=sound_alarm)
				t.deamon = True
				t.start()

		else:
			alert = 'normal'
			color = (0,255,0)
			ALARM_ON = False
	# show the prediction on the frame
		cv2.putText(frame, alert, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
					   1, color, 2, cv2.LINE_AA)


	# Show the final output
	cv2.imshow("Output", frame)

	if cv2.waitKey(1) == ord('q'):
		break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
