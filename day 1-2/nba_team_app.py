import html

import altair as alt
import pandas as pd
import streamlit as st
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams

from nba_skill_matcher import (
    FEATURE_COLUMNS,
    find_closest_players,
    format_importance_table,
    format_match_table,
    load_or_train_model,
    load_players,
)
from nba_team_simulator import load_simulator


st.set_page_config(
    page_title='NBA Game Simulator',
    page_icon='🏀',
    layout='wide',
    initial_sidebar_state='collapsed',
)

simulator = load_simulator()

NBA_NAVY = '#0B1F41'
NBA_RED = '#C8102E'
TEAM_COLORS = {
    'ATL': ('#C8102E', '#FDB927'), 'BOS': ('#007A33', '#BA9653'),
    'BKN': ('#111111', '#A7A9AC'), 'CHA': ('#1D1160', '#00788C'),
    'CHI': ('#CE1141', '#111111'), 'CLE': ('#6F263D', '#FFB81C'),
    'DAL': ('#00538C', '#B8C4CA'), 'DEN': ('#0E2240', '#FEC524'),
    'DET': ('#C8102E', '#1D42BA'), 'GSW': ('#1D428A', '#FFC72C'),
    'HOU': ('#CE1141', '#111111'), 'IND': ('#002D62', '#FDBB30'),
    'LAC': ('#1D428A', '#C8102E'), 'LAL': ('#552583', '#FDB927'),
    'MEM': ('#12173F', '#5D76A9'), 'MIA': ('#98002E', '#F9A01B'),
    'MIL': ('#00471B', '#EEE1C6'), 'MIN': ('#0C2340', '#78BE20'),
    'NOP': ('#0C2340', '#C8102E'), 'NYK': ('#006BB6', '#F58426'),
    'OKC': ('#007AC1', '#EF3B24'), 'ORL': ('#0077C0', '#C4CED4'),
    'PHI': ('#006BB6', '#ED174C'), 'PHX': ('#1D1160', '#E56020'),
    'POR': ('#E03A3E', '#111111'), 'SAC': ('#5A2D81', '#63727A'),
    'SAS': ('#111111', '#C4CED4'), 'TOR': ('#CE1141', '#A1A1A4'),
    'UTA': ('#002B5C', '#F9A01B'), 'WAS': ('#002B5C', '#E31837'),
}


@st.cache_data(show_spinner=False)
def all_team_data():
    return {team['full_name']: team for team in teams.get_teams()}


@st.cache_data(show_spinner=False)
def roster_names(team_name):
    team = simulator.find_team(team_name)
    roster = simulator.get_roster(team['id'])
    return roster['PLAYER'].sort_values().tolist()


def team_logo_url(team):
    return f"https://cdn.nba.com/logos/nba/{team['id']}/primary/L/logo.svg"


def player_headshot_url(player_id):
    return f'https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png'


def resolve_player_id(mvp):
    player_id = mvp.get('Player ID')
    if player_id is not None:
        try:
            return int(player_id)
        except (TypeError, ValueError):
            pass

    player_name = str(mvp.get('Player', '')).replace('*', '').strip()
    matches = nba_players.find_players_by_full_name(player_name)
    if matches:
        return matches[0]['id']

    return None


def inject_theme(primary, accent):
    st.markdown(
        f"""
        <style>
        :root {{ --nba-primary: {primary}; --nba-accent: {accent}; }}
        .stApp {{
            background:
                radial-gradient(circle at 12% 8%, {accent}35 0, transparent 28%),
                radial-gradient(circle at 90% 20%, {primary}75 0, transparent 34%),
                linear-gradient(145deg, #050A14 0%, {primary} 52%, #050A14 100%);
            color: #FFFFFF;
        }}
        .stApp::before {{
            content: '';
            position: fixed; inset: 0; pointer-events: none; opacity: .10;
            background-image: repeating-linear-gradient(120deg, transparent 0 38px, #fff 39px 40px);
        }}
        [data-testid='stHeader'] {{ background: transparent; }}
        [data-testid='stToolbar'] {{ right: 1rem; }}
        .block-container {{ max-width: 1200px; padding-top: 1.8rem; padding-bottom: 4rem; }}
        h1, h2, h3 {{ color: #fff !important; letter-spacing: -.02em; }}
        p, label, .stCaption {{ color: #EDF2F7 !important; }}
        .hero {{
            border: 1px solid #ffffff35; border-radius: 28px; padding: 28px 34px;
            background: linear-gradient(120deg, #050A14E8, {primary}DD);
            box-shadow: 0 22px 60px #0008; margin-bottom: 24px; overflow: hidden;
        }}
        .eyebrow {{ color: {accent}; font-weight: 800; letter-spacing: .18em; font-size: .78rem; }}
        .hero-title {{ font-size: clamp(2.35rem, 6vw, 5rem); font-weight: 950; line-height: .92; margin: 10px 0; }}
        .hero-copy {{ max-width: 680px; color: #DCE4EF; font-size: 1.05rem; }}
        .team-card, .preview-card, .result-card, [data-testid='stExpander'] {{
            background: #07101FD9; border: 1px solid #ffffff2C; border-radius: 22px;
            box-shadow: 0 14px 35px #0005;
        }}
        .team-card {{ min-height: 220px; display: flex; align-items: center; justify-content: center; padding: 18px; }}
        .team-card img {{ width: 150px; height: 150px; object-fit: contain; filter: drop-shadow(0 12px 15px #0008); }}
        .preview-card {{ padding: 22px; text-align: center; margin: 12px 0 22px; }}
        .preview-matchup {{ display:flex; align-items:center; justify-content:center; gap:22px; flex-wrap:wrap; }}
        .preview-team {{ font-weight: 900; font-size: 1.35rem; }}
        .preview-team img {{ width: 82px; height: 82px; object-fit:contain; display:block; margin:auto; }}
        .vs {{ width: 54px; height: 54px; border-radius:50%; display:grid; place-items:center; background:{accent}; color:#07101F; font-weight:950; }}
        .expected-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:18px; }}
        .expected-item {{ background:#ffffff10; border-radius:14px; padding:12px; color:#DCE4EF; }}
        .expected-item strong {{ display:block; color:#fff; margin-top:3px; }}
        .winner-banner {{
            padding: 20px 24px; border-radius: 20px; margin: 8px 0 22px;
            background: linear-gradient(110deg, {primary}, #07101F); border: 1px solid {accent};
            box-shadow: 0 0 34px {accent}45;
        }}
        .winner-kicker {{ color:{accent}; letter-spacing:.15em; font-weight:900; font-size:.76rem; }}
        .winner-name {{ font-size:2rem; font-weight:950; }}
        .mvp-card {{ background:#07101FD9; border:1px solid {accent}; border-radius:24px; padding:20px; }}
        div[data-testid='stButton'] button {{
            width:100%; min-height:58px; border:0; border-radius:16px; color:#07101F;
            font-size:1.08rem; font-weight:950; background:linear-gradient(90deg, {accent}, #FFFFFF);
            box-shadow:0 10px 26px #0007; transition:.18s ease;
        }}
        div[data-testid='stButton'] button:hover {{ transform:translateY(-2px); color:#07101F; }}
        [data-baseweb='select'] > div {{ background:#07101FEF; border-color:#ffffff40; border-radius:14px; color:#fff; }}
        [data-testid='stMetric'] {{ background:#07101FD9; border:1px solid #ffffff2C; padding:16px; border-radius:18px; }}
        [data-testid='stDataFrame'] {{ border-radius:18px; overflow:hidden; border:1px solid #ffffff2C; }}
        [data-testid='stButtonGroup'] {{
            position: relative; z-index: 10; padding: 8px;
            background: #050A14E8; border: 1px solid #ffffff32; border-radius: 16px;
            width: fit-content; margin: 3.25rem auto 20px; box-shadow: 0 10px 28px #0008;
            backdrop-filter: blur(16px);
        }}
        [data-testid='stButtonGroup'] button {{ background: #FFFFFF10 !important; color: #FFFFFF !important; }}
        [data-testid='stButtonGroup'] button[data-selected='true'] {{ background: {accent} !important; color: #07101F !important; }}
        .match-card {{
            background: linear-gradient(130deg, #07101FEF, {primary}CC); border: 1px solid {accent};
            border-radius: 24px; padding: 24px; box-shadow: 0 16px 38px #0007; margin: 10px 0 22px;
        }}
        .match-rank {{ color:{accent}; font-weight:900; letter-spacing:.14em; font-size:.76rem; }}
        .match-name {{ font-size:2.2rem; font-weight:950; margin:.2rem 0; }}
        @media (max-width: 760px) {{ .expected-grid {{ grid-template-columns:1fr 1fr; }} .hero {{padding:22px;}} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def team_picker_card(label, options, default_team, key):
    selected = st.selectbox(label, options, index=options.index(default_team), key=key)
    team = all_team_data()[selected]
    st.markdown(
        f'<div class="team-card"><img src="{team_logo_url(team)}" alt="{html.escape(selected)} logo"></div>',
        unsafe_allow_html=True,
    )
    return selected


def injury_checkboxes(team_name, key_prefix):
    st.markdown(f'#### {team_name}')
    st.caption('Select anyone who should sit out.')
    injured_players = []
    for player in roster_names(team_name):
        if st.checkbox(player, key=f'{key_prefix}_{player}'):
            injured_players.append(player)
    return injured_players


def team_box_score(box_score, team_abbreviation):
    team_stats = box_score[box_score['Team'] == team_abbreviation].copy()
    team_stats = team_stats.rename(columns={'MIN': 'Minutes', 'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists'})
    return team_stats[['Player', 'Minutes', 'Points', 'Rebounds', 'Assists']]


def styled_box_score(box_score, team_color):
    return (
        box_score.style
        .format({'Minutes': '{:.1f}', 'Points': '{:.0f}', 'Rebounds': '{:.0f}', 'Assists': '{:.0f}'})
        .set_properties(**{
            'background-color': '#07101F',
            'color': '#F7FAFC',
            'border-color': '#FFFFFF22',
        })
        .set_properties(subset=['Player'], **{'font-weight': '700', 'color': team_color})
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', team_color), ('color', '#FFFFFF'), ('font-weight', '800')]},
            {'selector': 'td', 'props': [('border-bottom', '1px solid #FFFFFF18')]},
        ])
    )


def player_stat_chart(box_score, stat, team_color):
    display_name = {'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists'}[stat]
    chart_data = box_score[['Player', stat]].sort_values(stat, ascending=False)
    chart_height = max(280, len(chart_data) * 34)

    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=8, height=20)
        .encode(
            x=alt.X(f'{stat}:Q', title=None, axis=alt.Axis(grid=True, tickMinStep=1)),
            y=alt.Y('Player:N', title=None, sort='-x', axis=alt.Axis(labelLimit=150)),
            color=alt.value(team_color),
            tooltip=[alt.Tooltip('Player:N'), alt.Tooltip(f'{stat}:Q', title=display_name)],
        )
        .properties(
            height=chart_height,
            background='#07101F',
            padding={'left': 16, 'right': 16, 'top': 16, 'bottom': 16},
        )
        .configure_view(fill='#07101F', stroke='#FFFFFF24', strokeWidth=1, cornerRadius=12)
        .configure_axis(
            domain=False,
            gridColor='#FFFFFF18',
            labelColor='#E8EEF7',
            labelFontSize=12,
            labelFontWeight=600,
            tickColor='#FFFFFF28',
            titleColor='#FFFFFF',
        )
    )


@st.cache_data(show_spinner=False)
def skill_player_data():
    return load_players()


@st.cache_resource(show_spinner=False)
def skill_matcher_model():
    return load_or_train_model(skill_player_data())


def render_team_simulator():
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">NBA MATCHUP LAB • {simulator.season}</div>
            <div class="hero-title">SIMULATE<br>THE NIGHT.</div>
            <div class="hero-copy">Choose the matchup, account for injuries, and generate a complete projected final with a winner, MVP, and player box scores.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    team_options = sorted(all_team_data())
    picker_left, picker_right = st.columns(2, gap='large')
    with picker_left:
        first_team = team_picker_card('Home team', team_options, 'Toronto Raptors', 'first_team')
    with picker_right:
        second_team = team_picker_card('Away team', team_options, 'Boston Celtics', 'second_team')

    if first_team == second_team:
        st.error('Pick two different teams to create a matchup.')
        return

    first_team_data = all_team_data()[first_team]
    second_team_data = all_team_data()[second_team]
    st.markdown(
        f"""
        <div class="preview-card">
            <div class="eyebrow">EXPECTED OUTPUT PREVIEW</div>
            <div class="preview-matchup">
                <div class="preview-team"><img src="{team_logo_url(first_team_data)}">{html.escape(first_team)}</div>
                <div class="vs">VS</div>
                <div class="preview-team"><img src="{team_logo_url(second_team_data)}">{html.escape(second_team)}</div>
            </div>
            <div class="expected-grid">
                <div class="expected-item">Projected score<strong>— : —</strong></div>
                <div class="expected-item">Winner<strong>After simulation</strong></div>
                <div class="expected-item">Game MVP<strong>After simulation</strong></div>
                <div class="expected-item">Player output<strong>Full box scores</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander('Injury report — optional'):
        injury_left, injury_right = st.columns(2)
        with injury_left:
            first_team_injuries = injury_checkboxes(first_team, 'first_team_injury')
        with injury_right:
            second_team_injuries = injury_checkboxes(second_team, 'second_team_injury')

    if st.button('🏀  SIMULATE MATCHUP', type='primary'):
        with st.spinner('Pulling NBA data and running the matchup...'):
            st.session_state.simulation_result = simulator.simulate(
                first_team, second_team, first_team_injuries, second_team_injuries,
            )
        st.rerun()

    result = st.session_state.get('simulation_result')
    if not result:
        return

    winner = result['winner']
    st.markdown(
        f"""
        <div class="winner-banner">
            <div class="winner-kicker">SIMULATED WINNER • THEME ACTIVATED</div>
            <div style="display:flex;align-items:center;gap:18px;margin-top:8px">
                <img src="{team_logo_url(winner)}" style="width:82px;height:82px;object-fit:contain">
                <div class="winner-name">{html.escape(winner['full_name'])}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_left, score_right, winner_col = st.columns(3)
    score_left.metric(result['team_one']['full_name'], result['team_one_score'])
    score_right.metric(result['team_two']['full_name'], result['team_two_score'])
    winner_col.metric('Winning margin', f"{abs(result['team_one_score'] - result['team_two_score'])} pts")

    mvp = result['mvp']
    mvp_player_id = resolve_player_id(mvp)
    st.markdown('## Game MVP')
    mvp_image, mvp_details = st.columns([1, 2], vertical_alignment='center')
    with mvp_image:
        if mvp_player_id is not None:
            st.image(player_headshot_url(mvp_player_id), use_container_width=True)
        else:
            st.markdown('<div class="mvp-card" style="font-size:6rem;text-align:center">🏀</div>', unsafe_allow_html=True)
    with mvp_details:
        st.markdown(f'### {mvp["Player"]}')
        mvp_one, mvp_two, mvp_three = st.columns(3)
        mvp_one.metric('PTS', int(mvp['PTS']))
        mvp_two.metric('REB', int(mvp['REB']))
        mvp_three.metric('AST', int(mvp['AST']))

    st.markdown('## Projected player stats')
    table_tab, chart_tab = st.tabs(['Box score', 'Stat charts'])
    team_one_box = team_box_score(result['box_score'], result['team_one']['abbreviation'])
    team_two_box = team_box_score(result['box_score'], result['team_two']['abbreviation'])
    team_one_color = TEAM_COLORS.get(result['team_one']['abbreviation'], (NBA_NAVY, NBA_RED))[1]
    team_two_color = TEAM_COLORS.get(result['team_two']['abbreviation'], (NBA_NAVY, NBA_RED))[1]

    with table_tab:
        first_stats_col, second_stats_col = st.columns(2)
        with first_stats_col:
            st.markdown(f'### {result["team_one"]["full_name"]}')
            st.dataframe(styled_box_score(team_one_box, team_one_color), use_container_width=True, hide_index=True)
        with second_stats_col:
            st.markdown(f'### {result["team_two"]["full_name"]}')
            st.dataframe(styled_box_score(team_two_box, team_two_color), use_container_width=True, hide_index=True)

    with chart_tab:
        selected_stat = st.radio(
            'Choose a stat', ['PTS', 'REB', 'AST'],
            format_func=lambda stat: {'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists'}[stat],
            horizontal=True, key='player_chart_stat',
        )
        first_chart_col, second_chart_col = st.columns(2)
        with first_chart_col:
            st.markdown(f'### {result["team_one"]["full_name"]}')
            st.altair_chart(player_stat_chart(result['box_score'][result['box_score']['Team'] == result['team_one']['abbreviation']], selected_stat, team_one_color), use_container_width=True)
        with second_chart_col:
            st.markdown(f'### {result["team_two"]["full_name"]}')
            st.altair_chart(player_stat_chart(result['box_score'][result['box_score']['Team'] == result['team_two']['abbreviation']], selected_stat, team_two_color), use_container_width=True)


def render_player_matcher():
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">NBA PLAYER DNA • RANDOM FOREST MATCHING</div>
            <div class="hero-title">FIND YOUR<br>NBA TWIN.</div>
            <div class="hero-copy">Enter your per-game numbers and shooting percentages. The model compares your profile with thousands of NBA player seasons to find your closest statistical matches.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="preview-card">
            <div class="eyebrow">HOW IT WORKS</div>
            <div class="expected-grid">
                <div class="expected-item">1<strong>Enter your stats</strong></div>
                <div class="expected-item">2<strong>Model scores your game</strong></div>
                <div class="expected-item">3<strong>Features are weighted</strong></div>
                <div class="expected-item">4<strong>Closest players appear</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form('player_match_form'):
        st.markdown('## Your player profile')
        stat_one, stat_two, stat_three = st.columns(3)
        points = stat_one.number_input('Points per game', min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        rebounds = stat_two.number_input('Rebounds per game', min_value=0.0, max_value=40.0, value=5.0, step=0.5)
        assists = stat_three.number_input('Assists per game', min_value=0.0, max_value=30.0, value=4.0, step=0.5)
        shooting_one, shooting_two, shooting_three, minutes_col = st.columns(4)
        field_goal = shooting_one.number_input('Field-goal %', min_value=0.0, max_value=1.0, value=0.45, step=0.01, format='%.2f')
        three_point = shooting_two.number_input('Three-point %', min_value=0.0, max_value=1.0, value=0.35, step=0.01, format='%.2f')
        free_throw = shooting_three.number_input('Free-throw %', min_value=0.0, max_value=1.0, value=0.75, step=0.01, format='%.2f')
        minutes = minutes_col.number_input('Minutes per game', min_value=0.0, max_value=48.0, value=28.0, step=0.5)
        submitted = st.form_submit_button('🔎  FIND MY NBA MATCH', type='primary')

    if submitted:
        with st.spinner('Comparing your game with NBA player profiles...'):
            user_stats = pd.DataFrame([{
                'PTS.1': points, 'TRB.1': rebounds, 'AST.1': assists,
                'FG%': field_goal, '3P%': three_point, 'FT%': free_throw, 'MP.1': minutes,
            }])
            dataset = skill_player_data()
            model = skill_matcher_model()
            closest = find_closest_players(dataset, model, user_stats, count=10)
            st.session_state.player_match_result = {
                'score': float(model.predict(user_stats[FEATURE_COLUMNS])[0]),
                'matches': format_match_table(closest),
                'importance': format_importance_table(model),
            }

    match_result = st.session_state.get('player_match_result')
    if not match_result:
        return

    top_match = match_result['matches'].iloc[0]
    top_player_id = resolve_player_id({'Player': top_match['Player']})
    match_image, match_details = st.columns([1, 2], vertical_alignment='center')
    with match_image:
        if top_player_id is not None:
            st.image(player_headshot_url(top_player_id), use_container_width=True)
        else:
            st.markdown('<div class="match-card" style="font-size:6rem;text-align:center">🏀</div>', unsafe_allow_html=True)
    with match_details:
        st.markdown(
            f"""
            <div class="match-card">
                <div class="match-rank">YOUR CLOSEST NBA MATCH</div>
                <div class="match-name">{html.escape(str(top_match['Player']))}</div>
                <div>{html.escape(str(top_match['Similarity']))} similarity • Model score {match_result['score']:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    matches_tab, model_tab = st.tabs(['Closest players', 'What the model values'])
    with matches_tab:
        st.dataframe(match_result['matches'], use_container_width=True, hide_index=True)
    with model_tab:
        st.dataframe(match_result['importance'], use_container_width=True, hide_index=True)


saved_result = st.session_state.get('simulation_result')
active_mode = st.session_state.get('active_site_mode', 'Team Simulator')
mode_choice = st.segmented_control(
    'Choose experience',
    ['Team Simulator', 'Find My NBA Match'],
    default=active_mode,
    key='site_mode_selector',
    label_visibility='collapsed',
    width='stretch',
)
selected_mode = mode_choice or active_mode
st.session_state.active_site_mode = selected_mode

if selected_mode == 'Team Simulator' and saved_result:
    winner_abbreviation = saved_result['winner']['abbreviation']
    primary_color, accent_color = TEAM_COLORS.get(winner_abbreviation, (NBA_NAVY, NBA_RED))
else:
    primary_color, accent_color = NBA_NAVY, NBA_RED

inject_theme(primary_color, accent_color)

if selected_mode == 'Find My NBA Match':
    render_player_matcher()
else:
    render_team_simulator()
