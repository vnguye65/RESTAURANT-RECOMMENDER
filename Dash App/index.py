#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:11:55 2021

@author: ving2000
"""


import numpy as np
import dash_bootstrap_components as dbc
import dash  
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table 
from geopy.geocoders import Nominatim

from Vi import users, usercuisine,  ratings, rclusters
import Vi

external_stylesheets=[dbc.themes.BOOTSTRAP]

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,  external_stylesheets=external_stylesheets)
server = app.server


app.layout = html.Div(
    [  
   #------------------------------------------------------------
         html.H1("Restaurant Recommender", style={'text-align': 'center', 
                                            'font-family': 'cursive',
                                            'fontWeight': 'bold'}),
         html.Div(className="row", children = [
             
    html.Div([
        html.Label("I am a ", 
                    style = {'fontWeight': 'bold', 'text-align': 'left'}),
  
   dcc.RadioItems(id = 'user_type',
                options=[
        {'label': 'Returning user', 'value': 1},
        {'label': 'New User', 'value': 0}
    ],
    value=1, labelStyle={'display': 'block', 
                            "padding": "1px", "margin": "auto"},  style = {'text-align': 'left'}
),
   

    html.Div([
            html.Label('Enter your User ID'),
            html.Br(),
            
            dcc.Input(id = 'uid', style={
                    'width': '50%',
                    'border': '2px',
                    'line-height': '25px'}),
            html.Br(),
            
            html.Button('SUBMIT', id = 'old_submit', style = {'fontWeight': 'bold',
                    'color': 'black',
                    'backgroundColor': 'orange',
                     'height': '40px',
                     'width': '100px'}),
            html.Br(),
            
            html.Button('Clear', id = 'old_clear')
            
            ], id = 'userID'),
    
    html.Label(id = 'label'),
    
   html.Div(id = 'intermediate-value', style = {'display': 'none'}),
   html.Br(),
   
   html.Div([

       
    html.H4('Answer a few questions', style = {'text-align': 'center'}),
    html.Br(),
    
   html.Label('Enter your address (format: Street, City, State, Country'),
   dcc.Input(id='address',
                  style={
                    'width': '100%',
                    'border': '0px',
                    'line-height': '25px'
                  }),
     
    
    html.Label("1. Are you a smoker?", 
                    style = {'fontWeight': 'bold', 'text-align': 'left'}),
  
      
   dcc.RadioItems(id = 'smoker',
                options=[
        {'label': 'Yes', 'value': 1},
        {'label': 'No', 'value': 0}
    ],
    value=None, labelStyle={'display': 'block', 
                            "padding": "1px", "margin": "auto"},  style = {'text-align': 'left'}
),
   
     html.Br(),
     
     html.Label("2. What is your drink level?", 
                    style = {'fontWeight': 'bold','text-align': 'left'}),
    
    dcc.Slider(id = 'alcohol', min=0, max=1, step = 0.5, marks={
        0: 'alcohol-free',
        0.5: 'social drinker',
        1: 'casual drinker'
    },
    value=0),
     
                  
       html.Br(),
      html.Label("3. How do you dress?", 
                    style = {'fontWeight': 'bold', 'text-align': 'left'}),
    
    dcc.RadioItems(id = 'dress',
                options=[
        {'label': 'I like to dress professional when I go out', 'value': 1},
        {'label': 'I dress casually', 'value': 0}
    ],
    value=None, labelStyle={'display': 'block', 
                            "padding": "1px",  "margin": "auto"}, 
    style = {'text-align': 'left'}
),
                  
         html.Br(),  
  
      html.Label("4. Do you enjoy the company of others when eating out?", 
                    style = {'fontWeight': 'bold', 'text-align': 'left'}),

    dcc.RadioItems(id = 'ambience',
                options=[
        {'label': 'Yes. I like to go out with family/friends.', 'value': 1},
        {'label': 'No. I like to go all by myself.', 'value': 0}
    ],
    value=None, labelStyle={'display': 'block', 
                            "padding": "1px",  "margin": "auto"}, 
    style = {'text-align': 'left'}
),
    
     html.Br(),
     
     html.Label("6. What is your budget?", 
                    style = {'fontWeight': 'bold','text-align': 'left'}),
    
    dcc.Slider(id = 'budget', min=0, max=1, step = 0.5, marks={
        0: '$',
        .5: '$$',
        1: '$$$$$'
    },
    value=0),
     
     html.Br(),           
                  
     html.Label("6. Select your favorite cuisine(s)?", 
                    style = {'fontWeight': 'bold', 'text-align': 'left'}),
     
     
    
    dcc.Checklist(id = 'cuisine',
                options=[
        {'label': 'African', 'value': 'African'},
        {'label': 'Asian', 'value': 'Asian'},
        {'label': 'Cafe', 'value': 'Cafe'},
        {'label': 'Fast Food', 'value': 'Fast_Food'},
        {'label': 'Latin', 'value': 'Latin'},
        {'label': 'European/Western', 'value': 'European/Western'},
        {'label': 'Middle Eastern/Eastern European', 'value': 'Middle Eastern/Eastern Europe'},
        {'label': 'Anything that is under the category of food', 'value': 'No preference'},
    ],
    labelStyle={'display': 'block', 
                            "padding": "1px",  "margin": "auto"}, 
    style = {'text-align': 'left'}
),
    
    html.Br(),
    
    html.Button('SUBMIT', id = 'new_submit', style = {'fontWeight': 'bold',
                    'color': 'black',
                    'backgroundColor': 'orange',
                     'height': '40px',
                     'width': '100px'
                    }),
    html.Br(),
    html.Button('Clear', id = 'new_clear')
    
    ], style = {'display':'none'}, id = 'new_user')
     
     
     ], style={'width': '30%', 'display': 'inline-block',
               'margin': 'auto', 'border': '1px solid black', 'marginTop': 20,
               'backgroundColor': 'rgb(255, 229, 204)', 'padding': "10px", 
               'maxHeight': '900px', 'overflow': 'scroll', 'align': 'left'}),
               
    html.Div(
        [
    dcc.Loading(id="loading",
           type="default",
    children = html.Div([
    html.Div(id = 'recoms'),
    html.Div(id = 'old_recoms')
    ])
    )],
             style = {'width': '60%', 'align': 'right', 'display': 'inline-block', 'padding': "5px", 
                      'verticalAlign': 'middle', 'margin': 'auto', 'marginTop': 15})
    
       ]) 
])

#----------------------------------------------------------------------------------------------------
@app.callback(
    [Output(component_id='userID', component_property='style'),
     Output(component_id='new_user', component_property='style'),
     Output(component_id='label', component_property='children'),
     Output(component_id='intermediate-value', component_property='children')],
    [Input(component_id='user_type', component_property='value')]
)

def getuid (val):
    
    if val == 1:
        divstyle = {'display': 'inline-block'}
        style = {'display': 'none'}
        message = None
        uid = None
        
    else:
        num = sorted(list(users.index))[-1]
        uid = 'U' + str(int(num[1:]) + 1)
        message = f'Your UserID is {uid}'
        divstyle = {'display': 'none'}
        style = {'display': 'block'}
       
    return divstyle, style, message, uid



@app.callback(
    Output(component_id='old_recoms', component_property='children'),
    
    [ Input(component_id='old_submit', component_property='n_clicks'),
      Input(component_id='uid', component_property='value')
      ]
)

def GetSimilarRestaurants (old_button, old_uid):
    if old_button != None:
        ids = users.index.tolist()
        lat, long = users.loc[old_uid][:2]
        
        R = Vi.GetRMatrix2(ratings, rclusters)
        W, H = Vi.NMFModel(R, 19)
        R_preds = np.dot(W, H)
        recoms = Vi.Skyline(R_preds, ids, lat, long, old_uid)
        recoms['rating'] = recoms['rating'].apply(lambda x: round((x/2)*100, 1))
        
        content =  html.Div([
            dash_table.DataTable(
                columns=[
    {'name': 'Rank', 'id': 'rank'},
    {'name': 'Restaurant', 'id': 'name'},
    {'name': '% Match', 'id': 'rating'},
    {'name': 'Distance', 'id': 'distance'}], 
    
    data = recoms.to_dict('records'), 
    
    style_cell={'textAlign': 'center', 
                'lineHeight': '15px'},
    style_header={
        'backgroundColor': 'rgb(255, 229, 204)',
        'fontWeight': 'bold',
        'lineHeight': '40px',
        'font_size': '25px',
        'textAlign': 'center'
    },
    
    style_data_conditional=[
        {
            'if': {
                'filter_query': '{rank} <= 10',
            },
            'backgroundColor': '#D3D3D3',
            'fontWeight': 'bold'
        }],

    
    style_table={
       'width': '100%', 
       'height': '900px', 
       'overflow': 'scroll',
       'border': '2px black solid'
    
    })  
    ])
    
        return content
    



@app.callback(
    Output(component_id='recoms', component_property='children'),
    
    [Input(component_id='smoker', component_property='value'),
      Input(component_id='alcohol', component_property='value'),
      Input(component_id='dress', component_property='value'),
      Input(component_id='ambience', component_property='value'),
      Input(component_id='budget', component_property='value'),
      Input(component_id='cuisine', component_property='value'),
      Input(component_id='new_submit', component_property='n_clicks'),
      Input(component_id='intermediate-value', component_property='children'),
      Input(component_id='address', component_property='value')
      ]
)

def GetRecommendations (smoker, alcohol, dress, ambience,
                        budget, cuisine, new_button, uid, address):
    
    if new_button != None:
        
        locator = Nominatim(user_agent="myGeocoder")
        location = locator.geocode(address)
        lat = location.latitude
        long = location.longitude
    
        new_user_arr = np.array([[lat, long, smoker, alcohol, dress, ambience, budget]])
        
        ratings_w_user = Vi.GetNeighbors(ratings, users, usercuisine, uid, new_user_arr, cuisine)
        #updated_users = Vi.Update(new_user_arr, uid, users)
        ids = users.index.tolist()
        ids = ids + [uid]
        
        R = Vi.GetRMatrix(ratings_w_user, ids)
        W, H = Vi.NMFModel(R, 19)
        R_preds = np.dot(W, H)
        #recoms, fdf = Vi.Recommend(restaurants, R_preds, uid, ids, 20)
        recoms = Vi.Skyline(R_preds, ids, lat, long, uid)
        recoms['rating'] = recoms['rating'].apply(lambda x: round((x/2)*100, 1))
        
        content =  html.Div([
            dash_table.DataTable(
                columns=[
    {'name': 'Rank', 'id': 'rank'},
    {'name': 'Restaurant', 'id': 'name'},
    {'name': '% Match', 'id': 'rating'},
    {'name': 'Distance', 'id': 'distance'}], 
                
    style_cell={'textAlign': 'center',
                            'lineHeight': '15px'},
    style_header={
        'backgroundColor': 'rgb(255, 229, 204)',
        'fontWeight': 'bold',
        'lineHeight': '40px',
        'font_size': '25px',
        'textAlign': 'center',
    },
    
    style_data_conditional=[
        {
            'if': {
                'filter_query': '{rank} <= 10',
            },
            'backgroundColor': '#D3D3D3',
            'fontWeight': 'bold'
        }],
    
    
    data = recoms.to_dict('records'), 
    style_table={
       'width': '90%', 
       'height': '900px', 
       'overflow': 'scroll', 
      'border': '2px black solid'
    }
    )  
    ]) 
        return content
          
    
@app.callback(
    Output(component_id='old_submit', component_property='n_clicks'),
    
    [Input(component_id='old_clear', component_property='n_clicks')]
)    

def Clearuid (button):
    if button != None:
        reset = None
        return reset
    
    
@app.callback(
    Output(component_id='new_submit', component_property='n_clicks'),
    
    [Input(component_id='new_clear', component_property='n_clicks')]
)    

def Clearentries (button):
    if button != None:
        reset = None
        return reset
       
    
    
if __name__ == '__main__':
    app.run_server(debug=True)    
    
    
    
    
    
    
    
    
    
