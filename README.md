# famous-faces

famous faces is a python project created by Jason Chen, Luc Cote, Filip Kierzenka, and Connor Browder at the 2018 Blueprint Hackathon.

Its purpose is to help raise awareness of significant figures in science in a fun way by finding a look-alike famous scientist to the user.

### Implemenetation
The backend is built using flask and is communicated to through HTML post requests.
- The homepage sends a post request to the backend with an image field containing a jpg image of the user
- The backend responds with a string of text representing the location of the image of the scientist which the user matches

The look-alike scientist is found by using the python face-recognition library to generate a feature map for each of the famous scientists and then comparing those feature maps to one of the user with a nearest-neighbor approach.

See it in action [here](https://famousfaces.herokuapp.com/)!
