from pydantic import BaseSettings


class Settings(BaseSettings):
    MONGO_INITDB_ROOT_USERNAME='admin'
    MONGO_INITDB_ROOT_PASSWORD='password'

    DATABASE_URL= 'mongodb://admin:password@140.120.13.246:27017/'


settings = Settings()