# Flask App [minimal]

Make sure you have the following packages installed in your virtual environment:
- `flask`
- `flask_bootstrap`
- `opencv-python`

Our App is currently deployed [here](https://eface-mask-app.herokuapp.com)
## Deploy Locally

Run `python app.py` and navigate on your web browser to `http://<your-public-ip>:5000`.

## Deploy to Heroku
1. [Install Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Sign up to Heroku CLI
3. Run `cd poc_flask_app`
4. Run `git init`
5. Run `git add .`
6. Run `git commit -m "first commit"`
4. Run `heroku login`
5. Run `heroku create your-app-name`
6. Run `git push heroku master`
7. App will be deployed to http://your-app-name.herokuapp.com

## Scripts

`video_app.py` is a script to open a native camera application on your computer (not accesible via web) for development purposes.