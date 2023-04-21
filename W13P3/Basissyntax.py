# int
eenGetal1 = 15

# String
eenTekst = "Dit is een string"

# Is het voor jou op een bepaald moment toch belangrijk om het type van een bepaalde variabele te weten, dan kan je het commando type gebruiken.
print(type(eenTekst))

# Je kan in één regel verschillende waardes toekennen aan verschillende variabelen.
eenGetal2, eenTekst, nogEenGetal = 15, "Dit is een string", 58

# In String literals die worden afgelijnd door double quotes worden single quotes als karakters aanzien, en vice versa.
print("De term 'hedendaags' wordt vaak gebruikt")
print('De term "hedendaags" wordt vaak gebruikt')

# Net zoals in Java is het escape karakter een backslash \
print ("Zowel 'single quotes' als \"double quotes\" worden hier gebruikt")

# Je kan ongeveer dezelfde karakters escapen als in Java
print("Deze tekst \n gaat op de volgende lijn verder")

# Wanneer je een String literal over meerdere lijnen wilt spreiden, zet je je String tussen tripple ' of "
print ("""Deze tekst
 staat over
 meerdere lijnen""")

# Net zoals in Java bestaan er ook een heel aantal stringfuncties ter beschikking in Python.
# eenTekst.index()
# eenTekst.len()
# eenTekst.replace()

# Strings vergelijken met elkaar doe je in Python gewoon met ==
eenTekst = "test"

# Het concatenatieteken in Python is +
voornaam = "Maarten"
achternaam = "Heylen"
volledigeNaam = voornaam + " " + achternaam
print(volledigeNaam)

# Wanneer Strings geconcateneerd moeten worden met getallen, dan moeten de getallen eerst worden omgezet naar een string met het str commando.
uitvoer = "Uw score is " + str(596)

# Mogelijke waardes voor booleans zijn True en False
mijnBooleanWaarde = True

# Python Lists
mijnEersteLijst = [] # Een lege lijst
mijnTweedeLijst = [25,87,3,46]
mijnDerdeLijst = ["Maarten", "Jan", "Bert"]
mijnVierdeLijst = ["Ben", 57, 'Kurt', True] # Verschillende types elementen
mijnVijfdeLijst = ["An", mijnDerdeLijst] # Geneste lists

mijnLijst = [1,2,3,4,5]
# Specifieke elementen van een lijst manipuleren gebeurd op soortgelijke manier als bij arrays.
mijnLijst[2]  # 3° element van de list
mijnLijst[0] = 42
len(mijnLijst)  # Aantal elementen in de lijst (lengte)
mijnLijst.__len__() # Aantal elementen in de lijst (lengte)

# Je kan ook verschillende elementen uit de list tegelijk selecteren.
mijnLijst[-1] # Selecteert het laatste element van de list
mijnLijst[1:4] # Selecteert 3 elementen uit de list
mijnLijst[::2] # Selecteert elementen met een even index uit de list (stapgrootte 2)
mijnLijst[::-1] # Geeft de elementen in de list in achterwaardse volgorde

# Python Tuple
mijnEersteLijst = () # Een lege tuple
mijnTweedeLijst = (25, 87, 3, 46)
mijnDerdeLijst = ("Maarten", "Jan", "Bert")
mijnVierdeLijst = ("Ben", 57, 'Kurt', True) # Verschillende types elementen
mijnVijfdeLijst = ("An", mijnDerdeLijst) # Geneste tuples

# Python Dictionary (Map in Java) - Schrijven
punten = {"Jan": 15, "Ben": 19, "An": 16}
punten['Jan'] = 16
punten['Peter'] = 20

# Python Dictionary (Map in Java) - Lezen
print (punten['Jan'])
print (punten.keys())

# Python Dictionary (Map in Java) - functies
punten.keys()
punten.pop('Jan')
len(punten)
# If
inkomsten = 1
if inkomsten > 0:
 print ("U heeft winst gemaakt.")
 print ("Goed gedaan van u")
print ("Deze opdracht hoort niet meer bij de if-clause")

# If en else
inkomsten = 1
if inkomsten > 0:
 print("U heeft winst gemaakt.")
else:
 print("U heeft verlies gemaakt.")

# elseif
if inkomsten > 0:
 print("U heeft winst gemaakt.")
elif inkomsten == 0:
 print("Juist break even gedraaid.")
else:
 print("U heeft verlies gemaakt.")

# While
getal = 2002
while getal > 10:
 getal = getal - 1
 print(getal)

# For
# Teller van 0 tot 9
print ("ik tel tot tien (excl tien)")
for teller in range(0,10):
 print ("huidig getal: " + str(teller))

# Foreach: itereren door een list
studenten = ["An", "Ben", "Jan"]
for huidigeStudent in studenten:
 print ("Welkom" + huidigeStudent)

# Foreach: itereren door een tuple
studenten = ("An", "Ben", "Jan")
for huidigeStudent in studenten:
 print ("Welkom" + huidigeStudent)

# Test van print()
print("""
E     Eerlijk
L     Lief
I     Introvert
A     Actief
S     Slim""")

# 7 // 2 geeft 3, omdat 7 gedeeld door 2 geheel deelbaar is met rest 1, en we alleen het gehele deel willen

# 7 / 2 geeft 3.5, omdat 7 gedeeld door 2 geen geheel getal is

x = "10"
y = int(x)
print(y) # Output: 10

x = "10.5"
y = float(x)
print(y) # Output: 10.5

x = 10
y = str(x)
print(y) # Output: "10"

x = "hello"
y = list(x)
print(y) # Output: ["h", "e", "l", "l", "o"]

x = ["apple", "banana", "cherry"]
y = tuple(x)
print(y) # Output: ("apple", "banana", "cherry")

x = ["apple", "banana", "cherry"]
y = set(x)
print(y) # Output: {"apple", "banana", "cherry"}
