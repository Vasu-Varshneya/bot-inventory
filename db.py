import mysql.connector
from mysql.connector import pooling

# Connection pool to avoid reconnecting each request
connection_pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,
    pool_reset_session=True,
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port=4000,
    user="4FqQsswjPt3ZaUg.bot",   # 👈 your bot user
    password="password",          # 👈 your bot password
    database="test",
    ssl_ca=r"C:\tidb\isrgrootx1.pem"  # 👈 path to pem file
)

def get_connection():
    return connection_pool.get_connection()

