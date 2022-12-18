#!/usr/bin/env python
# coding: utf-8

# # Final Assignment 

# Take care, that collaboration in solving the assignment is not allowed and can lead to non-passing of the asssigment. I will check the solutions for similarities. Solve the tasks before October 30th 2022. You are only allowed to import the following modules:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import statsmodels.api as sm
import statsmodels.formula.api as smf

import pickle

from selenium import webdriver
import json


# ## 1. Data preprocessing and plots

# **1.1** Import 'players_21.csv' as a pandas data dataframe. Define a column ("year_born") that contains the year of birth of each player as an integer.

# In[2]:


df=pd.read_csv('players_21.csv')
df


# In[3]:


df['year_born'] = df['dob']
df['year_born'] = pd.to_datetime(df["year_born"]).dt.strftime("%Y%m%d").astype(int)
print( df['year_born'] )


# **1.2** Show the most valuable player for each nation ('nationality'). Display the position in the national team ('nation_position'), the name ('short_name'), and the market value in millions.

# In[4]:


df['market_value'] = df['value_eur'] / 10**6
df['market_value']

most_valuable_player = df[["short_name", "nationality", "nation_position","market_value"]].drop_duplicates(subset='nation_position', keep ='first').groupby("nation_position").max()
print(most_valuable_player)


# In[ ]:





# **1.3** Show all Clubs ('club_name') of the "German 1. Bundesliga". Sort them by thier total market value ('value_eur
# ')in descending order. Display the club name, the total market value and the
# size of the squad!

# In[5]:


# df_league=df_teams= df.groupby("league_name").head(10)
# df_league

# test = df.query('league_name == "German 1. Bundesliga"').groupby('club_name')
# test['league_name'].head(250000)

sum_df = df.query('league_name == "German 1. Bundesliga"').groupby('club_name')[['value_eur']].agg(['sum', 'count']).head(199)
result_df = sum_df['value_eur'].nlargest(n=10, columns='sum')
result_df['market_value'] = (result_df['sum'].astype(float)/1000000).astype(str) 
result_df['size_of_squad'] = result_df['count']
del result_df['sum']
del result_df['count']
result_df

#df_league = df.sort_values(['club_name']).groupby(['league_name'])
# test = df.query('league_name == "German 1. Bundesliga"').groupby(['league_name'])
# newdf = test['club_name','league_name','market_value','size_of_squad']
# newdf


# **1.4** Filter the dateset for the 1000 most valueable players. Create a new  Plot the minium, the mean, the 99%-quantile,  and the maximum value for each age ('age') group. (Replicate ``'age.pdf'.``)

# In[6]:


valuable_player1 = df.copy()
#valuable_player2 = valuable_player1[['marker_value','age']]
#df['market_value'] = df['value_eur'] / 10**6

valuable_player1['market_value'] = valuable_player1['value_eur'] / 10**6

#valuable_player1[['market_value', 'age']].head(1000)

players = valuable_player1[['market_value', 'age']].sort_values(by='market_value', ascending = False).head(1000)

#min 
player_min = players.groupby('age') ['market_value'].min()

#max 
player_max = players.groupby('age')['market_value'].max()

#average value 
player_mean = players.groupby('age')['market_value'].mean()

# quantile
player_quantile = players.groupby('age')['market_value'].quantile(.99)


#plotting
player_min.plot(label='min')
player_mean.plot(label='mean')
player_quantile.plot(label='quantile')
player_max.plot(label='max')

plt.xlabel('age')
plt.ylabel('Market_value in millions')
plt.legend(fontsize=10)
plt.show()


# ***1.5***  Print the the 3 most frequent jersey numbers ('team_jersey_number') for each team postion ('team_position').

# In[9]:


#test1 = df.groupby('team_position')
#test1[['team_position','team_jersey_number']].head(1000)
#collect_jer_num = test1['team_jersey_number'].mode()

#df_frequent= df[['team_position']].groupby(['team_jersey_number','team_position']).count(['team_jersey_number']).head(10)

df_frequent = df.groupby('team_position')['team_jersey_number'].value_counts().head(3)
df_frequent


# ***1.5*** Add lines of code in the following template to replicate ``'potential.pdf'``. Explain precisely the line where dot_color is determined and why this determination is not computational efficient.

# In[10]:


#create matrix and define dimenstion

#dim = [100, 100]

x_col = 100
x_row = 100
matrix = np.zeros((x_row, x_col), dtype=int)

for overall,potential in zip(df['overall'],df['potential']):
    # Increase the value in the matrix in ov row and the pot column by 1
    matrix[overall, potential] += 1



colors=[('w',0),('y',1),('c',5),('g',20),('b',50),('k',100)]
print(colors)

# Iterate over ov grid
overall = 0
for  row in matrix:
    ## Iterate over pot grid
    potential = 0
    for column in row:
        if column != 0:
            dot_color = [color for color,val in colors if val<= matrix[overall,potential]][-1]
#why
            ##Plot a dot with ov as x,pot as y and dot_color as color
            plt.scatter(overall, potential, color= dot_color, s=20)
        potential += 1
    overall += 1 
    
    #'SOlution is not sufficient because data points are mix with each other'
    
#Add the x and y label
plt.xlabel('overall')
plt.ylabel('potential')

#Save the figure
plt.savefig("potentail.pdf", format="pdf")

plt.show()


# ## 2. Regression

# ***2.1*** Delete all players where the value below the 25% quartile and drop all players that are not playing for a national team ('nation_position').

# In[12]:


df_filter=df[(df['value_eur']<0.25) & (df['nation_position'])!=0 ]

df_filter.head(100)


# ***2.2*** Regress the logarithmic player value on the overall strength ('overall') and potential ('potential') of the player.  Which player is the most overvalued (highest residual value)?

# In[13]:



over= df.copy()
# we need to non zero column for regression

actual_result = over[over['value_eur'] != 0]

overvaluedDf = actual_result[['short_name', 'value_eur', 'overall', 'potential' ]]

#overvaluedDf['value_eur'] = valuable_player1['value_eur'] / 10**6


#converting values 
overvaluedDf.value_eur = np.log(overvaluedDf.value_eur)


##fit the model
reg = smf.ols("value_eur ~ overall + potential", data=overvaluedDf).fit()

# now for prediction
predic = reg.predict()
overvaluedDf = overvaluedDf.assign(predicted_value = predic)
overvaluedDf = overvaluedDf.assign(residuals = overvaluedDf['value_eur'] - overvaluedDf['predicted_value'])

# highest residual
overvaluedDf.loc[overvaluedDf['residuals'].idxmax()]



# ***2.3*** Plot the residuals and logarithmic player value in scatter plot.

# In[ ]:





# ***2.4*** Create a single column for every postion in 'team_position', which is one if a player plays on that postion an zero in all other cases. Regress the logarithmic player value on age, the squared age, Body-Mass-Index and the team position dummies.  Hint: The BMI is ``weight / height**2``.

# In[14]:


position= actual_result.copy()

find = position[['short_name', 'age','weight_kg', 'height_cm', 'value_eur', 'team_position']]
find.head()

#converting values into log
find = find.assign(value_log = np.log(find.value_eur))

mass = find.set_index('short_name').team_position.str.split(',', expand=True).stack()
dummy = pd.get_dummies(mass).groupby(level=0).sum()

result = pd.merge(find, dummy, on="short_name")
result.head(30)


# In[ ]:


print(set(df['team_position']))


# In[15]:


position2 = result.copy()

regressor = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB + LCM + RB + RDM + LCB + CF + LW + CAM + LF + RF + RS + RW", data=position2).fit()

print(regressor.summary())
print(regressor.params)
print(position2.columns)


# ***2.5*** Use the same model as in the last task. Now estimate all possible models in which you omit one explanatory variable (position dummies count as single explanatory variables). Which has the highest and lowest influence on the R Squared? (Use a loop!)

# In[16]:


posdummies = position2.copy()
#according to last task we get
regression1 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB + LCM + RB + RDM + LCB + CF + LW + CAM + LF + RF + RS + RW", data=position2).fit()
regression2 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB + LCM + RB + RDM + LCB + CF + LW + CAM + LF + RF + RS", data=position2).fit()                      
regression3 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB + LCM + RB + RDM + LCB + CF + LW + CAM + LF + RF", data=position2).fit()
regression4 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB + LCM + RB + RDM + LCB + CF + LW + CAM + LF", data=position2).fit()
regression5 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB + LCM + RB + RDM + LCB + CF + LW + CAM", data=position2).fit()
regression6 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB + LCM + RB + RDM + LCB + CF + LW", data=position2).fit()
regression7 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB + LCM + RB + RDM + LCB + CF", data=position2).fit()
regression8 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB + LCM + RB + RDM + LCB", data=position2).fit()
regression9 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB + LCM + RB + RDM", data=position2).fit()
regression10 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB + LCM + RB", data=position2).fit()
regression11 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB + LCM", data=position2).fit()
regression12 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM + LWB", data=position2).fit()
regression13 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB + CM", data=position2).fit()
regression14 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM + CB", data=position2).fit()
regression15 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM + LDM", data=position2).fit()
regression16 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM + RM", data=position2).fit()
regression17 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB + LAM", data=position2).fit()
regression18 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK+ RCB", data=position2).fit()
regression19 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES+GK", data=position2).fit()
regression20 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB+RES", data=position2).fit()
regression21 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM+LB", data=position2).fit()
regression22 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB+CDM", data=position2).fit()
regression23 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM+RWB", data=position2).fit()
regression24 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST+RCM", data=position2).fit()
regression25 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB + ST", data=position2).fit()
regression26 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM + SUB", data=position2).fit()
regression27 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM + RAM", data=position2).fit()
regression28 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS+ LM ", data=position2).fit()
regression29 = smf.ols("value_eur ~ age + age**2 + weight_kg/height_cm**2 +LS", data=position2).fit()


#fitting the model 
#regression1.fit()
#regression2.fit()
#regression3.fit()
#regression4.fit()
#regression5.fit()

print('R-squared values are:')

r2_dict={
    'Regression1':  regression1.rsquared,
    'Regression2':  regression2.rsquared,
    'Regression3':  regression3.rsquared,
    'Regression4':  regression4.rsquared,
    'Regression5':  regression5.rsquared,
    'Regression6':  regression6.rsquared,
    'Regression7':  regression7.rsquared,
    'Regression8':  regression8.rsquared,
    'Regression9':  regression9.rsquared,
    'Regression10':  regression10.rsquared,
    'Regression11':  regression11.rsquared,
    'Regression12':  regression12.rsquared,
    'Regression13':  regression13.rsquared,
    'Regression14':  regression14.rsquared,
    'Regression15':  regression15.rsquared,
    'Regression16':  regression16.rsquared,
    'Regression17':  regression17.rsquared,
    'Regression18':  regression18.rsquared,
    'Regression19':  regression19.rsquared,
    'Regression20':  regression20.rsquared,
    'Regression21':  regression21.rsquared,
    'Regression22':  regression22.rsquared,
    'Regression23':  regression23.rsquared,
    'Regression24':  regression24.rsquared,
    'Regression25':  regression25.rsquared,
    'Regression26':  regression26.rsquared,
    'Regression27':  regression27.rsquared,
    'Regression28':  regression28.rsquared,
    'Regression29':  regression29.rsquared,
    
}

print(pd.Series(r2_dict).sort_values(ascending=True))


# ## 3. Password
# 
# Create a program that performs the following tasks: (You are only allowed to import time and pickle. Use ``password.py`` as a template.)
# 
# 1. When you start the program it aks you to enter a password. 
# 
# 2. The password is stored in a pickle file. If the password is incorrect, it should be requested again. If the password is correct, you can choose between the following options:
#     
#     2.1. P: Show the information which is stored in 'secret_file.txt'!
# 
#     2.2. C: Change the password.
# 
#     2.3. L: Lock the program. (Go back to password request.)
# 
#     2.4. X: Exit the program.
# 
# 3. If there is no pickle file with a password, it should ask you to set a new password and then store it into the pickle file.
# 
# 4. If one enters the wrong password six times in row, it should wait for 10 seconds until it shows the message. 
# 
# 5. It should check if the password has a minimum length of 10 and if consists of at least one letter and ask for confirmation. It should also print if the requirement is fulfilled.
# 
# 
# 
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time
from termcolor import colored

def set_password():
    while True:
        n_pass = input('Enter your password:\n')
        if len(n_pass) < 8:
            print('Atleast 8 character')
            continue
        else:
            print('Congratualations')
        if n_pass=="E":
            continue
        if n_pass != input('Confirm your password:\n'):
            print(colored('Password  must be equal to the previous password.','red'))
            continue
        pickle.dump(n_pass, open(r'D:\Study\TU Dortmund\python\password.py', 'wb'))
        print('Password successfully changed.')
        return n_pass

a=0
nflag = False

while True:
    try:
        password = pickle.load(open(r"D:\Study\TU Dortmund\python\password.py", "rb") )
    except:
        password = set_password()

    enter_password= input('Enter the password\n')


    if enter_password == password  :
        while True:
            condition = input('Enter "S" to show secret file, Enter "C" to change the password, Enter "L" to lock, Enter "X" to exit the programm.\n').upper()
            if condition == "S": 
                secret_file = open(r'D:\Study\TU Dortmund\python\secret_file.txt', 'r', encoding='utf8')
                secret_file11 = secret_file.read()
                print(secret_file11)
                secret_file.close()
                continue

            if condition == "C":
                password = set_password()

            if condition == "L":
                break
                
            if condition == "X":
                nflag = True
                break                   
                            
            print(colored(' not detected', 'red'))
        if nflag :
            break
    
    elif enter_password == "E":
        break        
    else:
        a += 1
        if a == 5 :
            time.sleep(10)
            a = 0
        print(colored('Password not correct!', 'red'))


# ## 4. Webscraping

# Create a program that takes the date in the format MM-DD as an input and queries all people on the English-language Wikipedia (for example https://en.wikipedia.org/wiki/July_12) that are born on that day and saves it in a JSON file. (Note: See ``'07-12.json'`` as an example file.)

# In[ ]:



import sys
get_ipython().system('{sys.executable} -m pip install selenium')
from selenium import webdriver


# In[ ]:


date='07-12'

month_list=['January','February','March','April','May','June','July','August','September','October','November','December']


# In[ ]:


caths = ['Events','Births','Deaths']
data=[]
cath_index=0
last_year=-10e10
for element in driver.find_elements_by_tag_name('li'):
    try:
        text=element.text
        if ' – '  in text:
            year= 
            year= -int(year[:year.find(' ')]) if 'BC' in year else int(year)
            person= 
            discription = 
            
                        
            if year<last_year-1000:
                cath_index+=1
            last_year=year


            if caths[cath_index]=='Births':
                data.append()
    except Exception as e:
        print(e)
        pass
        
json.dump(data,open(,'w'))

