from app import app

# This is what gunicorn will look for
application = app

if __name__ == '__main__':
    application.run()