from pydantic_settings import BaseSettings, SettingsConfigDict

class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='POSTGRES_',
        env_file=".env" # if os.getenv("RUNNING_IN_DOCKER") != "1" else None
    )
    USER: str
    PASSWORD: str
    DB: str
    HOST: str
    PORT: int

    @property
    def db_url(self):
        return (
            f"postgresql://{self.USER}:{self.PASSWORD}"
            f"@{self.HOST}:{self.PORT}/{self.DB}"
        )

