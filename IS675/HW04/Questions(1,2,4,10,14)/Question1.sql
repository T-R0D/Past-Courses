--Question1: Which clients are located in Reno? 
 --List the ClientID, ClientName, ClientCity, ClientState, ClientZip and Phone.

 SELECT	ClientID 'Clientid',
		ClientName,
		ClientCity,
		ClientState,
		ClientZip,
		Phone  'ClientPhone'
 FROM Client
 WHERE ClientCity = 'Reno';