import json
import tweepy as tw
from urllib.request import urlopen
from tweepy import auth
from geopy.geocoders import Nominatim
from datetime import datetime

# variables for keys
api_key = "<key>"
api_key_secret = "<key>"
access_token = "<token>"
access_token_secret = "<token>"
latitude = 33.91
longitude = -98.49

# auth for connecting twitter api
auth = tw.OAuthHandler(api_key,api_key_secret)
auth.set_access_token(access_token,access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# open weather api
key = '<key>'
url = 'https://api.openweathermap.org/data/2.5/onecall?lat='+ str(latitude) + '&lon=' + str(longitude) + '&appid=' + key + '&units=imperial'

# get a response and update data variable
response = urlopen(url)
data = json.loads(response.read())

temperature = str(int(round(data['current']['temp'])))
# get date and time 

time = str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

# get a location using our latitude and longitude
geolocator = Nominatim(user_agent="test")
location = geolocator.reverse(str(latitude) + "," + str(longitude))
address = location.raw['address']
city = address.get('city','')

# Post a tweet
api.update_status(time + '\nCurrent Temperature in ' + str(city) + ' is ' + temperature + '°F')
