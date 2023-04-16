import altair as alt
import pandas as pd
import panel as pn
from vega_datasets import data
import os
pn.extension('vega')

# Make sure that the Panel version is ≥ 0.14.0.
pn.__version__

# Specify the directory path
dir_path = "/Users/flickr-xc/Library/Mobile Documents/com~apple~CloudDocs/DSBA Courses/Data Visualization/assignment 3/DV"
# Specify the file name
file_1= "raw_titles.csv"
file_2= "show.csv"
file_3= "movie.csv"
file_4= "df_actor_info.csv"
file_5= "df_actor_detail.csv"
file_6= "sorted_cooperation_df.csv"


# TITLE
description_title = pn.pane.Markdown("""
    <span style='font-family: Open Sans; font-size: 40px; font-weight: bold;'>
    Netflix Silhouette
""")


# 1.Country map

# Create a markdown object with the description text
description = pn.pane.Markdown("""
    <span style='font-family: Open Sans; font-size: 20px; font-weight: bold;'>
    Getting to know Netflix? Let's start with its expansion across the world.
    Use the dropdown menu to select the type of content and the slider to choose the year. 
    Hover over a country to see the name of the country, the count of the selected type, and the year.
""")


basedir = os.path.join(dir_path, file_1)

data1 = pd.read_csv(basedir)
data1.drop(columns = 'index', inplace = True)
data1.sort_values('release_year', inplace = True)
# Filter data to keep rows with non-null values these 3 columns
data1 = data1.dropna(subset=['production_countries', 'type','release_year'])

# the first year a country produced Netflix content, the content name and the according imdb_score
data1 = data1.assign(production_countries=data1['production_countries'].str.split(','))
data1 = data1.explode('production_countries')
data1['production_countries'] = data1['production_countries'].str.replace('[^a-zA-Z]', '')
year_country = pd.concat([data1.release_year,data1.production_countries,data1.type],axis=1)
countryexplode = year_country.explode("production_countries")

countryeplode_num =countryexplode.groupby(['production_countries','release_year','type'])["production_countries"]\
          .agg([('num_productions','count')]).reset_index()\
          .sort_values(by=["release_year",'production_countries'] , ascending=[True,True])

countryeplode_num =  countryeplode_num.drop(index =countryeplode_num[(countryeplode_num.production_countries == '')].index.tolist() )
countryeplode_num.reset_index(inplace = True,drop = True)

# Group data by 'production_countries' and 'release_year' columns and count the number of shows in each group
country_year_counts = countryeplode_num.groupby(['production_countries','release_year','type'])['num_productions'].agg('sum').reset_index(name='Count')
country_year_counts.columns = ['Country','Year','Type','Count']

country_year_counts[country_year_counts['Country']=='US']

#prepare data for base map
world = data.world_110m.url
world_topo = data.world_110m()
world_topo.keys()
world_topo['objects'].keys()

#prepare Country Codes
code_df = pd.read_json(
    "https://raw.githubusercontent.com/alisle/world-110m-country-codes/master/world-110m-country-codes.json"
)
# Add a missing country
singapore_df = pd.DataFrame(data=[{"code": "Singapore", "id": 702, "name": "Singapore"}])
code_df = pd.concat([code_df, singapore_df])

# Merge netflix country data with country code
code_df2 = code_df.set_index("code")
country_year_counts2 = country_year_counts.set_index("Country")
df = country_year_counts2.join(code_df2).reset_index()

base_map=alt.Chart(alt.topo_feature(world, 'countries')).mark_geoshape( stroke='#d8e3e7', 
                                  strokeWidth=0.3,
                                  color='#f9f9f9'
                                  # color='#d8e3e7'
                                  )
# base_map

count_max = df['Count'].max()
count_min = df['Count'].min()
chart = alt.layer(
    base_map,

    alt.Chart(alt.topo_feature(world, 'countries')).mark_geoshape( 
    stroke='#FFFFF', 
    strokeWidth=0.25
    ).transform_lookup(
        lookup='id', from_=alt.LookupData(data=df, key='id', fields=['Count','Year','Type','index','name'])
    ).encode(
        alt.Color('Count:Q',
                  scale=alt.Scale(clamp=True,domain=[count_min, count_max]), 
                  legend=alt.Legend()),
        tooltip=['name:N', 'Year:N','Count:Q','Type:N'],
    ).project(
        type='equalEarth'
    )
    ).properties(
        width=1000,
        height=500
    ).configure_view(
        stroke=None
    )


type_selector = pn.widgets.Select( value='type', width=150, options=['MOVIE', 'SHOW'])
year_slider = pn.widgets.IntSlider(name='Year',width=850, start=int( 2000
                                # df['Year'].min()
                                ),
                                   end=int(df['Year'].max()), step=1, value=int(df['Year'].max()))


def update_chart(event):
    selected_type = type_selector.value
    selected_year = year_slider.value
    filtered_df = df[(df['Year'] == selected_year) & (df['Type'] == selected_type)]


    # filtered data
    chart = alt.layer(
        base_map,
        alt.Chart(alt.topo_feature(world, 'countries')).mark_geoshape().transform_lookup(
            lookup='id', from_=alt.LookupData(data=filtered_df, key='id', fields=['Count', 'Year', 'Type', 'index', 'name'])
        ).encode(
            
            alt.Color('Count:Q',
                      scale=alt.Scale(clamp=True,domain=[count_min, count_max]), 
                      legend=alt.Legend()),
            tooltip=['name:N', 'Year:N', 'Count:Q', 'Type:N'],
            
        ).project(
            type='equalEarth'
        )
    ).properties(
        width=1000,
        height=500
    ).configure_view(
        stroke=None
    )
    
    # Update the chart in the Panel layout
    layout1[2] = chart


                               
# Attach the callback function
type_selector.param.watch(update_chart, 'value')
year_slider.param.watch(update_chart, 'value')

# Create a panel layout with the selector and the chart
layout1 = pn.Column(
    '## PART1: Netflix Expansion by Year and Country',
    pn.Row(type_selector,
    year_slider),
    chart,
    description
)


# 2.Genre
import matplotlib.pyplot as plt
import seaborn as sns
from altair import expr, datum
import numpy as np
import panel as pn
import pandas as pd
import altair as alt

# prepare data

description2 = pn.pane.Markdown("""
    <span style='font-family: Open Sans; font-size: 20px; font-weight: bold;'>
    Great, Let's look at the score distribution on Netflix in terms of distinctive genres. What a huge video streaming family!
    Pleas use the dropdown menu to select the type of content and zoom in and out to have a better visualizaton!
""")

df3 = pd.read_csv(basedir)
df3 = df3.set_index('index')


df_alt = df3.assign(production_countries=df3['production_countries'].str.split(','))
df_alt = df_alt.explode('production_countries')
df_alt['production_countries'] = df_alt['production_countries'].str.replace('[^a-zA-Z]', '')

df_alt = df_alt.assign(genres=df_alt['genres'].str.split(','))
df_alt = df_alt.explode('genres')
df_alt['genres'] = df_alt['genres'].str.replace('[^,a-zA-Z]', '')

df_alt = df_alt[0:15000]

df4 = df_alt.drop_duplicates(subset='id', keep='first')
df4 = df4.sort_values('imdb_votes', ascending=False).reset_index(drop=True)
df4 = df4.reset_index(drop=False)
df4 = df4[['index','id','title','type','genres','imdb_score','imdb_votes']]

color = alt.Color('genres:N')

pn.extension('vega')


def plot_scatter(dataframe):
    # 1. Copy the scatter plot and modify encodings based on params
    brush = alt.selection_interval(name = 'brush',encodings = ['x'])
    multi = alt.selection_multi(name='multi', fields=['genres'])
    points = alt.Chart(dataframe).mark_point().encode(
        alt.X('index:Q', title='movie/show'),
        alt.Y('imdb_score:Q',
            title='imdb_score',
            scale=alt.Scale(domain=[-1, 10])
        ),
        color=alt.condition(brush,color,alt.value('lightgrey'))
    ).properties(
        width=400,
        height=310
    ).add_selection(brush)

    # 2. Retuen the Vega pane of the Altair chart
    return pn.pane.Vega(points, debounce=100)

def plot_bar(selection,dataframe):
    # 1. Filter the penguin data frame based on selection
    if selection is None or len(selection) == 0:
        df_plot_bar = dataframe
    else:
        query = ' & '.join(
            f'{crange[0]} <= `{col}` <= {crange[1]}'
            for col, crange in selection.items())
        df_plot_bar = dataframe.query(query)

    # 2. Create the bar chart
    multi = alt.selection_multi(name = 'multi',fields=['genres'])

    brush = alt.selection_interval(name = 'brush',encodings = ['x'])

    bars = alt.Chart(df_plot_bar).mark_bar().encode(
        x='mean(imdb_votes):Q',
        y='genres:N',
        color=alt.condition(multi,color,alt.value('lightgrey')),
        tooltip=['mean(imdb_votes):Q', 'genres:N','Type:N'],
    ).properties(width=400, height=310).add_selection(multi)
    # Add multiple selection to the bar chart


    # Filter the data when user draws brush on the bubble chart
    # bars = bars.transform_filter(brush)
    return bars


# movie/show selection
type = pn.widgets.Select(value='type', width=150, options=['MOVIE', 'SHOW'])


pane = plot_scatter(df4)

# 3. Update the panel when the inputs are changed
def update(event):
    select_type = type.value
    filtered_df = df4[(df4['type'] == select_type)]
    pane = plot_scatter(filtered_df)
    layout2[2][0][0] = pane
    layout2[2][0][1] = pn.bind(plot_bar, pane.selection.param.brush,filtered_df)

# 2. Watch the parameter change and call update function

type.param.watch(update, 'value')


# 4. Create the panel with inputs and all panes

layout2 = pn.Column(
    '## PART2: Netflix Number of Votes Explorer',
    type,
    pn.Column(
        pn.Row(pane, pn.bind(plot_bar, pane.selection.param.brush,df4))
        ),
    description2
)

# layout2


# 3.IMDB_score

"""
1.   Load dataset
2.   import Altair
3. plot
"""

# max row > 5000
alt.data_transformers.disable_max_rows()

# show
# Load dataset
basedir_2 = os.path.join(dir_path, file_2)
basedir_3 = os.path.join(dir_path, file_3)

show = pd.read_csv(basedir_2)
movie = pd.read_csv(basedir_3)

show = show.loc[:, ['id', 'title', 'type','genre_split', 'imdb_score']]
source_show = show.rename(columns={'genre_split': 'genre'})

movie = movie.loc[:, ['id', 'title', 'type','genre_split', 'imdb_score']]
source_movie = movie.rename(columns={'genre_split': 'genre'})

# exclude null
source_show = source_show.dropna(subset=['genre'])
source_movie = source_movie.dropna(subset=['genre'])

# plot - show
scat_show = alt.Chart(source_show).mark_point(size = 50).encode(
    y = alt.Y('mean(imdb_score):Q', scale=alt.Scale(domain=(4,8)), title='Average Imdb Score'),
    x = alt.X('count(id)', title='Count of show'),
    color = 'genre',
    size = alt.Size('count(id)'),
    tooltip = ['genre:N', 'mean(imdb_score):N']
).properties(
    width=400,
    height=400
)


scat_movie = alt.Chart(source_movie).mark_point(size = 50).encode(
    y = alt.Y('mean(imdb_score):Q', scale=alt.Scale(domain=(4,8)),title='Average Imdb Score'),
    x = alt.X('count(id)', title='Count of movie'),
    color = 'genre',
    size = alt.Size('count(id)'),
    tooltip = ['genre:N', 'mean(imdb_score):N']
).properties(
    width=400,
    height=400
)

description3 = pn.pane.Markdown("""
    <span style='font-family: Open Sans; font-size: 20px; font-weight: bold;'>
    Wonder about each genre’s performance?
    Drama takes the lead in both movie and show category. We all love dramas!! 
""")
                                
                
layout3=pn.Column(
    '## PART3: Netflix Imdb Score',
    pn.Row(scat_movie,
    scat_show),
    description3
    )
layout3


# 4.Actor
import altair as alt
import pandas as pd
import panel as pn
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from altair.expr import datum

pn.extension("material")  # apply the 'elegant' theme

# max row > 5000
alt.data_transformers.disable_max_rows()

# Load dataset
basedir_4 = os.path.join(dir_path, file_4)
basedir_5 = os.path.join(dir_path, file_5)
basedir_6 = os.path.join(dir_path, file_6)

df_actor_info = pd.read_csv(basedir_4)
df_actor_detail = pd.read_csv(basedir_5)
sorted_cooperation_df = pd.read_csv(basedir_6)

# define the widgets
name_input = pn.widgets.AutocompleteInput(name='Name', options=list(df_actor_info['name'].unique()), case_sensitive=False, placeholder='Enter name...')
type_input = pn.widgets.Select(name='Type', options=['movie', 'show'], value='movie', width=200)

# define the function that updates the plot
def update_plot(name, type):
    filtered_df = df_actor_info[(df_actor_info['name'].str.lower() == name.lower()) & (df_actor_info['type'].str.lower() == type)]

    if not filtered_df.empty:
        total_votes = int(filtered_df['total_votes'].iloc[0])
        average_score = round(filtered_df['average_score'].iloc[0], 2)
        total_numbers = filtered_df['total_numbers'].iloc[0]

        total_votes_pane = pn.pane.HTML(
          f'<h2 style="color:#0077C0;font-size:40px;font-family: Verdana">{total_votes:,}</h2><br><h4 style="color:#0077C0;font-family: Verdana">Total Votes</h4>',
          margin=(0, 0, 20, 0))
        average_score_pane = pn.pane.HTML(
          f'<h2 style="color:#FFB000;font-size:40px;font-family: Verdana">{average_score}</h2><br><h4 style="color:#FFB000;font-family: Verdana">Average Score</h4>',
          margin=(0, 20, 20, 0))
        total_numbers_pane = pn.pane.HTML(
          f'<h2 style="color:#DC267F;font-size:40px;font-family: Verdana">{total_numbers}</h2><br><h4 style="color:#DC267F;font-family: Verdana">Number of Works</h4>',
          margin=(0, 20, 20, 0))

        result = pn.Row(pn.Column(total_votes_pane, average_score_pane, total_numbers_pane),
                        sizing_mode='stretch_both', margin=(0, 20))

        return result
        
    else:
        return pn.pane.HTML('<h3>No data found.</h3>')

# define the function that updates the table
def update_table(name, type):
    filtered_df = df_actor_detail[(df_actor_detail['name'].str.lower() == name.lower()) & (df_actor_detail['type'].str.lower() == type)]

    if not filtered_df.empty:
        table_rows = []
        for index, row in filtered_df.iterrows():
            title = row['title']
            type = row['type'].capitalize()
            genres = row['genres']
            release_year = row['release_year']
            imdb_score = row['imdb_score']
            imdb_votes = int(row['imdb_votes'])
            row_html = f"""
            <div style="background-color:#F0F8FF;border: 2px solid #E65100;border-radius:5px;padding:10px;margin-bottom:10px">
                <h3 style="color:#E65100;font-family: Verdana">{title}</h3>
                <table style="margin-left:10px">
                    <tr>
                        <td style="font-family: Verdana;font-size:10px"><b>Type:</b></td>
                        <td style="font-family: Verdana;font-size:10px">{type}</td>
                    </tr>
                    <tr>
                        <td style="font-family: Verdana;font-size:10px"><b>Genre:</b></td>
                        <td style="font-family: Verdana;font-size:10px">{genres}</td>
                    </tr>
                    <tr>
                        <td style="font-family: Verdana;font-size:10px"><b>Year of Release:</b></td>
                        <td style="font-family: Verdana;font-size:10px">{release_year}</td>
                    </tr>
                    <tr>
                        <td style="font-family: Verdana;font-size:10px"><b>IMDB Score:</b></td>
                        <td style="font-family: Verdana;font-size:10px">{imdb_score}</td>
                    </tr>
                    <tr>
                        <td style="font-family: Verdana;font-size:10px"><b>IMDB Votes:</b></td>
                        <td style="font-family: Verdana;font-size:10px">{imdb_votes:,}</td>
                    </tr>
                </table>
            </div>
            """

            table_rows.append(row_html)

        result = pn.Column(*table_rows, sizing_mode='stretch_both', margin=(0, 20))
        return result
        
    else:
        return pn.pane.HTML('<h3>No data found.</h3>')


# Set the default theme for Altair charts
alt.themes.enable('fivethirtyeight')
# Define a custom CSS style for the text pane
text_style = 'font-family: Arial; font-size: 14px; color: #333; margin: 5px 0;'

def update_cooperate(name, type):
    filtered_df = sorted_cooperation_df[(sorted_cooperation_df['name_x'].str.lower() == name.lower())]

    if not filtered_df.empty:
        # Filter the top 10 partners with the largest number of cooperations
        df_top_10 = filtered_df.nlargest(20, 'cooperations')

        # Define the color scheme for the bars
        color_scale = alt.Scale(range=['#E50914', '#F5F5F5', '#B81D24', '#D81E26', '#F0343F', '#E50914', '#F9AA33', '#F5F5F5', '#B81D24', '#D81E26']
)

        # define x_axis
        x_axis = alt.Axis(
        title='Number of Cooperations',
        labelExpr="datum.label == floor(datum.label) ? format(datum.label, 'd') : ''",
        grid=False,
        tickCount=6
        )
        # Create the bar chart using Altair
        bars = alt.Chart(df_top_10).mark_bar(size=15, opacity=0.7).encode(
            x=alt.X('cooperations', title='Number of Cooperations', axis=x_axis),
            y=alt.Y('name_y', title='Cooperation Partner', sort=alt.EncodingSortField(field='cooperations', op='sum', order='descending')),
            color=alt.Color('cooperations', scale=color_scale, legend=None),
            tooltip=[alt.Tooltip('name_y:N', title='Partner Name'), alt.Tooltip('cooperations', title='Number of Cooperations')]
        ).properties(
            height=400,
            width=600,
            title=f'Cooperation Partners of {name.title()}',
            background='#f5f5f5'
        ).interactive()

        text_pane = pn.pane.HTML(margin=(20, 10), style={'background-color': '#f9f9f9', 'border': '1px solid #cccccc', 'border-radius': '5px', 'padding': '10px'})

        # Create the layout and return it
        layout = pn.Row(bars, text_pane, margin=(20, 20), background='#f9f9f9')
        return layout
        
    else:
        return pn.pane.HTML('<h3>No data found.</h3>')
    

description4 = pn.pane.Markdown("""
    <span style='font-family: Open Sans; font-size: 20px; font-weight: bold;'>
    Who is your favorite Actor or Director? Wanna know him/her more? Check this out!!
    Please type and choose the name as well as the type, We give you everything about him/her on Netflix!
""")
                                
description5 = pn.pane.Markdown("""
    <span style='font-family: Open Sans; font-size: 20px; font-weight: bold;'>
   WOW, Christopher Nolan and Cillian Murphy are Golden Partners!
""")

    
# define the panel app
@pn.depends(name_input.param.value, type_input.param.value)
def panel_app(name, type):
    if name and type:
        # create the two panes side by side using Tabs
        return pn.Tabs(('Overview', update_plot(name, type)), \
                       ('Into Detail', update_table(name, type)),\
                       ('Best Partners',update_cooperate(name, type)))
    else:
        return pn.pane.HTML('<h3>Please select name and type.</h3>')
    


# show the panel app
layout4 =pn.Column(
    '## PART4: Actors On Netflix',
    pn.Row(name_input, type_input),
    panel_app,
    description4,description5
)

layout4


layout_final = pn.Column(
    description_title,
    layout1,
    layout2,
    layout3,
    layout4
    )

layout_final.servable()
