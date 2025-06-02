# 🎬 The Duration Dilemma

**Case #2: Investigating the relationship between movie length and IMDb ratings**

Hey there! This is my data analysis project where I tackle that age-old question: do longer movies actually deserve higher ratings? We've all sat through those epic 3-hour films, but are they really better than their shorter counterparts? Let's find out!

## What's This All About?

I analyzed the IMDB 5000 Movie Dataset to see if there's any real correlation between how long a movie is and how highly it's rated. Here's what I looked into:

- The actual correlation between runtime and IMDb scores (spoiler: it exists, but it's weak!)
- Which duration categories get the best ratings (short, medium, or long?)
- Where the sweet spot is for movie length
- How movies are distributed across different durations

## Getting Started

### What You'll Need

- Python 3.6+
- Some packages (just run `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scipy
  - streamlit
  - plotly

### Setting Up

1. Clone or download this repo
2. Install the packages:
   ```
   pip install -r requirements.txt
   ```
3. Make sure the `movie_metadata.csv` file is in the main folder

### Running the Analysis Script

Just run:
```
python movie_analysis.py
```

This will:
- Clean the data and do all the calculations
- Save cool visualizations in the `plot` folder
- Give you all the insights about movie durations and ratings

### Check Out The Interactive Dashboard!

The fun part - an interactive dashboard where you can explore the data yourself:
```
streamlit run app.py
```

The dashboard lets you:
- Filter movies by ratings
- Highlight specific duration categories
- Find your own "sweet spot" for movie length
- Play with the visualizations to discover patterns

## Key Findings

So what did I discover? Here's the scoop:

- There's a weak positive correlation (0.26) between movie duration and ratings
- Long movies (>150 min) average 7.44/10, while short ones (<90 min) average only 6.15/10
- The magical sweet spot seems to be 180-190 minutes with an impressive 7.71/10 average!
- Most movies (80.2%) fall in the medium category (90-150 minutes)

## Dashboard Features

The interactive dashboard is where things get really interesting:
- Rating filters to focus on just the movies you're interested in
- Category highlighting to compare different duration groups
- A "Sweet Spot Finder" to discover optimal runtimes
- Customizable histograms to visualize the distribution
- All the stats you could want about movie durations

## The Verdict

After crunching all the numbers, here's the deal: longer movies do tend to get slightly better ratings on average, BUT the relationship is pretty weak. Quality storytelling matters way more than runtime!

As the old Hollywood saying goes: "No good movie is too long, and no bad movie is short enough."

---

Written by manish paneru | Data Analyst 