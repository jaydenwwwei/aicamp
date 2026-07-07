from nba_team_simulator import simulate_game


def parse_injuries(value):
    return [player.strip() for player in value.split(',') if player.strip()]


first_team = input('First team: ')
second_team = input('Second team: ')
first_team_injuries = parse_injuries(
    input(f'Injured players for {first_team}, separated by commas. Leave blank for none: ')
)
second_team_injuries = parse_injuries(
    input(f'Injured players for {second_team}, separated by commas. Leave blank for none: ')
)

simulate_game(first_team, second_team, first_team_injuries, second_team_injuries)
