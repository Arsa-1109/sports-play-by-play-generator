"""Built-in curated and procedurally varied corpus of sports news and play-by-play commentary."""
from __future__ import annotations

import random

_BASE_ARTICLES: list[str] = [
    (
        "CHAMPIONS CUP FINAL: LATE STRIKER HEROICS SEAL DRAMATIC COMEBACK\n"
        "Ninety minutes on the clock, and the striker ghosted past the back four to meet a whipped cross from the right flank. "
        "His first-time volley kissed the inside of the far post and nestled into the back of the net as the stadium erupted. "
        "The goalkeeper had no chance against the pace and swerve of the finish from twelve yards out.\n"
        "Earlier in the second half, the visitors controlled midfield possession with crisp one-touch passing and relentless pressing. "
        "Their holding midfielder broke up three dangerous counterattacks before threading a diagonal ball behind the fullback. "
        "However, missed chances inside the six-yard box kept the margin narrow, leaving the door ajar for a stoppage-time equalizer and winner. "
        "When the fourth official raised the board for five added minutes, the home crowd rose in unison and roared their side forward. "
        "A corner kick swung deep to the penalty spot was nodded down, setting up the decisive strike that clinched the trophy."
    ),
    (
        "PREMIER DERBY PLAY-BY-PLAY: COUNTERATTACKING MASTERCLASS IN THE RAIN\n"
        "Kickoff under heavy floodlights and slick turf immediately favored quick vertical passes through the channels. "
        "In the fourteenth minute, the winger nutmegged his marker near the touchline, drove into the penalty area, and squared a low pass across the face of goal. "
        "The center-forward timed his run to perfection, sliding in ahead of the center-back to tap home the opening goal.\n"
        "Play-by-play in the sixty-eighth minute: a crunching tackle in midfield wins possession cleanly. "
        "The playmaker spins away from two challenges and launches a thirty-yard through ball down the left wing. "
        "The overlapping fullback cushions the ball on his chest, cuts inside onto his stronger right foot, and curls a dipping shot over the outstretched fingertips of the keeper into the top corner. "
        "In stoppage time, the goalkeeper preserved the clean sheet with a stunning reflex save low to his left off a deflected free kick."
    ),
    (
        "WORLD QUALIFIER REPORT: PENALTY SHOOTOUT NERVE AND TACTICAL DISCIPLINE\n"
        "After one hundred and twenty grueling minutes ended deadlocked at one goal apiece, the tie went to a nerve-shredding penalty shootout. "
        "Both defensive lines held compact shapes throughout extra time, blocking shots from the edge of the box and clearing every dangerous set piece. "
        "The manager introduced fresh legs on the wings in the hundredth minute, nearly finding a winner when a curling effort rattled the crossbar.\n"
        "From the spot, the captain stepped up first and buried a thunderous penalty into the roof of the net. "
        "The opposing keeper guessed right on the third kick, diving full stretch to parry a low drive away from danger. "
        "With the fifth and final kick, the young midfielder sent the goalkeeper the wrong way with a cool side-footed finish into the bottom left corner to spark wild celebrations."
    ),
    (
        "NBA CONFERENCE FINALS: BUZZER-BEATING THREE-POINTER STUNS THE ARENA\n"
        "With four seconds remaining on the game clock and the score tied at 108, the point guard brought the ball across half-court against tight full-court pressure. "
        "A high ball screen from the power forward forced a defensive switch, leaving the seven-foot center isolated on the perimeter. "
        "Two quick crossovers created half a step of separation behind the three-point arc. "
        "He rose cleanly over the outstretched contest and drained a step-back three-pointer as the red lights flashed around the backboard.\n"
        "Throughout the fourth quarter, both teams traded momentum in a breathless display of shot-making and defensive grit. "
        "The home squad dominated the glass with fourteen offensive rebounds, turning second-chance opportunities into crucial points in the paint. "
        "A chasedown block at the rim with forty seconds left swung possession back, setting the stage for the unforgettable game-winner."
    ),
    (
        "BASKETBALL PLAY-BY-PLAY: FAST BREAKS, ALLEY-OOPS, AND CLUTCH FREE THROWS\n"
        "Third quarter underway and the pace is relentless. The shooting guard jumps the passing lane for a clean steal at the top of the key and sprints down the floor on a three-on-one fast break. "
        "Instead of forcing a contested layup, he lobs a soft pass off the glass for the trailing forward, who catches it in mid-air and slams home a thunderous two-handed alley-oop dunk.\n"
        "Timeout called by the visiting coach after a 12-0 run flips a seven-point deficit into a five-point lead. "
        "Out of the huddle, the offense runs a crisp pick-and-roll action on the weak side. "
        "The guard snakes around the screen, draws two defenders in the lane, and kicks out to the corner sniper, who splashes a catch-and-shoot triple nothing but net. "
        "Down the stretch, intentional fouls send the veteran guard to the free-throw line, where he sinks four consecutive clutch free throws to ice the victory."
    ),
    (
        "TEST CRICKET THRILLER: FINAL OVER DRAMA AND REVERSE SWING MASTERY\n"
        "Six runs needed off the final over with only one wicket standing, and the fast bowler steamed in from the pavilion end with a reversing old ball. "
        "The first delivery was a searing yorker clocked at ninety miles an hour, jammed out cleanly into the covers for a single. "
        "With five required from four balls, the set batter backed away to the leg side and carved a fierce square cut past backward point to the boundary rope for four.\n"
        "Tension gripped the ground as two dot balls followed—a sharp bouncer ducked under and a swinging slower delivery beaten past the outside edge. "
        "With one run to tie and two to win off the last ball, the batter lofted an inside-out drive over extra cover, sprinting back for the second run as the throw from the deep missed the stumps by inches."
    ),
    (
        "ONE-DAY INTERNATIONAL COMMENTARY: CHASE MASTERCLASS UNDER LIGHTS\n"
        "Chasing 312 under lights on a true batting surface, the openers laid a brisk foundation during the mandatory powerplay. "
        "Crisp cover drives pierced the infield, while effortless flicks off the pads punished anything straying onto leg stump. "
        "When the leg-spinner came into the attack in the fifteenth over, he struck twice in five balls—first trapping the left-hander plumb in front with a disguised googly, then deceiving the new batter in flight for a sharp stumping.\n"
        "Rebuilding through the middle overs, the captain rotated the strike with nimble singles and quick twos between the wickets before unleashing in the death overs. "
        "A towering pull shot sailed ten rows back over deep midwicket for six, followed by a delicate scoop over fine leg that brought up a match-winning century off ninety-four deliveries."
    ),
    (
        "WORLD SERIES BASEBALL: WALK-OFF HOME RUN IN THE BOTTOM OF THE NINTH\n"
        "Bottom of the ninth inning, two outs, runners on first and second, and a full count on the cleanup hitter. "
        "The closer stared in for the sign, shook off a slider, and fired a ninety-eight-mile-per-hour four-seam fastball on the inner half. "
        "The slugger turned on it with thunderous bat speed, sending a towering fly ball deep toward the right-field foul pole.\n"
        "The right fielder drifted back to the warning track, leaped at the wall, and watched it sail over the bullpen into the roaring crowd for a walk-off three-run home run. "
        "Earlier, a pitching duel had kept both lineups quiet through seven innings, featuring twelve strikeouts, painted corners on the outside edge, and a sparkling 6-4-3 double play to escape a bases-loaded jam."
    ),
    (
        "BASEBALL PLAY-BY-PLAY: STOLEN BASES, CURVEBALLS, AND BULLPEN GRIT\n"
        "Leading off the sixth inning, the speedster worked a patient seven-pitch walk after fouling off three tough two-strike breaking balls. "
        "Taking a generous lead off first base, he broke on the pitcher's first move and slid headfirst under the tag to steal second base cleanly. "
        "Two pitches later, a sharp line-drive single into center field brought him racing around third to slide safely across home plate ahead of the throw.\n"
        "In the eighth, the setup reliever entered with the tying run at third and struck out the side on sharp downward curveballs that dropped out of the strike zone."
    ),
    (
        "GRAND SLAM TENNIS FINAL: FIVE-SET MARATHON ON CENTER COURT\n"
        "Serving for the championship at five-four in the deciding fifth set, the top seed faced double break point after two unforced errors off the forehand wing. "
        "Stepping up to the baseline, he pounded a 132-mile-per-hour ace down the T, then erased the second break point with a brave serve-and-volley drop shot that died on the grass.\n"
        "At deuce, a grueling twenty-eight-shot rally unfolded across every corner of the court—deep topspin groundstrokes pinned both players behind the baseline until a breathtaking backhand passing shot down the line kissed the chalk. "
        "On championship point, a kicking second serve forced a short return, allowing a crisp inside-out forehand winner to clinch the title after four hours and forty minutes."
    ),
    (
        "TENNIS PLAY-BY-PLAY: TIEBREAK TENSION AND NET-RUSHING BRILLIANCE\n"
        "Fourth-set tiebreak locked at six-all as both players refuse to blink on serve. "
        "The challenger steps inside the baseline to crush a flat backhand return deep at his opponent's shoelaces, forcing a weak half-volley that he puts away with an overhead smash.\n"
        "Serving for the set at seven-six, he mixes pace brilliantly—sliding a wide slice serve into the ad court before charging the net and feathering a crosscourt volley winner out of reach. "
        "Throughout the match, high first-serve percentages and aggressive court positioning kept pressure squarely on the returner."
    ),
    (
        "PREMIER FOOTBALL PLAY-BY-PLAY: COUNTERATTACKING GOAL FROM CORNER CLEARANCE\n"
        "Eighty-fourth minute and the visitors defend a crowded six-yard box against an inswinging corner kick. "
        "The towering center-back heads the danger clear to the edge of the area, where the box-to-box midfielder traps the bouncing ball and slips a pass wide.\n"
        "Three touches later, the pacey striker races past the last defender, rounds the rushing goalkeeper, and rolls the ball into the empty net to silence the home end."
    ),
    (
        "BASKETBALL FOURTH-QUARTER CRUNCH: PICK-AND-ROLL READS AND DEFENSIVE STOPS\n"
        "Down two points with twenty-four seconds left, the home team calls timeout to draw up a late-game set. "
        "Inbounding from the sideline, the point guard comes off a sturdy screen at the top of the key and snakes into the paint.\n"
        "Drawing help from the weak-side corner, he whips a crisp skip pass to the waiting shooter, who squares up and drills the go-ahead three-pointer with three seconds remaining."
    ),
    (
        "ASHES TEST MATCH COMMENTARY: REVERSE SWING SPELL UNDER CLOUD COVER\n"
        "Dark clouds roll across the ground as the fast bowler charges in from the southern end with a scuffed fifty-over ball. "
        "Targeting the stumps at ninety miles an hour, he shapes a late inswinger back through the gate to clip the top of off stump.\n"
        "Two balls later, a sharp lifting delivery kisses the shoulder of the bat and carries cleanly to gully, sparking a dramatic collapse before tea."
    ),
    (
        "MAJOR LEAGUE PLAY-BY-PLAY: BASES-LOADED DOUBLE PLAY AND CLUTCH SAVE\n"
        "Top of the eighth inning with the bases loaded and one out, and the reliever enters needing a ground ball. "
        "He pounds a heavy sinker onto the inside corner, jamming the batter into a sharp chopper toward shortstop.\n"
        "The shortstop ranges left, flips cleanly to second base for one, and the second baseman fires across his body to first in time to complete the inning-ending 6-4-3 double play."
    ),
    (
        "WIMBLEDON QUARTERFINAL: SERVE-AND-VOLLEY MASTERCLASS ON GRASS\n"
        "Serving at five-all in the third-set tiebreak, the defending champion lands a biting kick serve out wide to pull the returner off the court. "
        "Following the serve straight to the net, he absorbs a low passing shot with a cushioned drop volley that dies just inches over the tape.\n"
        "One point later, an unreturnable ace down the center line seals the set in emphatic style."
    ),
]


def _generate_football_articles(rng: random.Random) -> list[str]:
    titles = [
        "CHAMPIONS CUP PLAY-BY-PLAY",
        "PREMIER LEAGUE MATCH REPORT",
        "DERBY NIGHT COMMENTARY",
        "KNOCKOUT TIE RECAP",
    ]
    clocks = [
        "Ninety minutes on the clock, and the striker",
        "Seventy-eighth minute on the clock, and the striker",
        "Deep into stoppage time, and the striker",
        "Early in the second half, and the winger",
        "Eighty-fourth minute under the floodlights, and the forward",
        "Sixty-second minute at the stadium, and the midfielder",
    ]
    buildups = [
        "ghosted past the back four to meet a whipped cross from the right flank",
        "timed a sharp diagonal run behind the center-back to latch onto a through ball",
        "dropped into midfield space, spun past two defenders, and drove toward the penalty box",
        "took a crisp first touch near the edge of the box and shifted onto his stronger foot",
        "sprinted down the left touchline to collect a long switch of play from the captain",
    ]
    finishes = [
        "His first-time volley kissed the inside of the far post and nestled into the back of the net.",
        "He curled a low shot around the rushing goalkeeper into the bottom corner as the crowd erupted.",
        "He buried a thunderous half-volley off the crossbar and down across the goal line.",
        "He slipped a composed finish between the goalkeeper and the near post from twelve yards out.",
        "He nodded a towering header back across goal into the top corner past outstretched fingertips.",
    ]
    responses = [
        "Earlier in the half, the visitors controlled midfield possession with crisp one-touch passing and relentless pressing.",
        "The holding midfielder broke up two dangerous counterattacks before launching a quick pass out wide.",
        "The goalkeeper kept his side alive moments earlier with a stunning reflex save low to his left.",
        "Both defensive lines held compact shapes throughout the match, blocking shots from the edge of the box.",
        "When the fourth official raised the board for added minutes, the home fans rose and roared their side forward.",
    ]
    endings = [
        "A corner kick swung deep to the penalty spot sealed a dramatic comeback victory.",
        "The clean sheet and late winner moved the club top of the standings on a memorable night.",
        "The referee blew the full-time whistle seconds later as players celebrated in front of the away end.",
        "Tactical discipline in midfield and clinical finishing inside the six-yard box decided the contest.",
        "It was a fearless attacking display that turned a tight tactical battle into a classic cup victory.",
    ]

    articles: list[str] = []
    seen: set[str] = set()
    while len(articles) < 34:
        title = rng.choice(titles)
        clock = rng.choice(clocks)
        buildup = rng.choice(buildups)
        finish = rng.choice(finishes)
        resp = rng.choice(responses)
        end = rng.choice(endings)
        article = f"{title}: {clock.split(',')[0].upper()}\n{clock} {buildup}. {finish}\n{resp} {end}"
        if article not in seen:
            seen.add(article)
            articles.append(article)
    return articles


def _generate_basketball_articles(rng: random.Random) -> list[str]:
    titles = [
        "NBA PLAYOFF PLAY-BY-PLAY",
        "CONFERENCE FINALS RECAP",
        "FOURTH-QUARTER CLUTCH REPORT",
        "ARENA NIGHT COMMENTARY",
    ]
    openers = [
        "With four seconds remaining on the game clock and the score tied,",
        "With twelve seconds left in the fourth quarter and trailing by one,",
        "Midway through the third quarter as the arena pace surged,",
        "Down two points with twenty-four seconds remaining on the clock,",
        "Opening possession of overtime in front of a roaring home crowd,",
    ]
    actions = [
        "the point guard brought the ball across half-court against tight full-court pressure and attacked a high ball screen.",
        "the shooting guard jumped the passing lane for a clean steal at the top of the key and sprinted down the floor.",
        "the floor general snaked around a sturdy pick-and-roll in the lane and drew two defenders toward the paint.",
        "the All-Star forward came off a staggered double screen on the wing and squared his shoulders behind the arc.",
        "the point guard drove past his defender on the right elbow and whipped a crisp skip pass to the open corner shooter.",
    ]
    shots = [
        "Two quick crossovers created half a step of separation before he drained a step-back three-pointer at the buzzer.",
        "He lobbed a soft pass off the glass for the trailing forward, who slammed home a thunderous two-handed dunk.",
        "He rose cleanly over the outstretched contest and splashed a catch-and-shoot triple nothing but net.",
        "He knocked down a silky fadeaway jumper from the elbow and drew the foul for a decisive three-point play.",
        "He spun smoothly in the lane and finished a high-arcing floater over the seven-foot center at the rim.",
    ]
    extras = [
        "The home squad dominated the glass with fourteen offensive rebounds, turning second-chance looks into crucial points in the paint.",
        "A chasedown block at the rim moments earlier swung momentum back and set up the unforgettable game-winner.",
        "Timeout called by the visiting coach after a 12-0 run flipped a seven-point deficit into a five-point lead.",
        "Down the stretch, intentional fouls sent the veteran guard to the free-throw line, where he sank four straight clutch free throws.",
        "Aggressive defensive rotations and sharp weak-side help protected the rim through every late possession.",
    ]

    articles: list[str] = []
    seen: set[str] = set()
    while len(articles) < 34:
        article = (
            f"{rng.choice(titles)}: LATE-GAME EXECUTION\n"
            f"{rng.choice(openers)} {rng.choice(actions)} {rng.choice(shots)}\n"
            f"{rng.choice(extras)}"
        )
        if article not in seen:
            seen.add(article)
            articles.append(article)
    return articles


def _generate_cricket_articles(rng: random.Random) -> list[str]:
    titles = [
        "TEST CRICKET PLAY-BY-PLAY",
        "ODI CHASE COMMENTARY",
        "T20 DEATH-OVER REPORT",
        "STADIUM MATCH SUMMARY",
    ]
    openers = [
        "Six runs needed off the final over with only one wicket standing, and the fast bowler",
        "Twelve runs required from the last six deliveries under bright floodlights, and the pacer",
        "Opening spell on a lively surface with cloud cover overhead, and the bowler",
        "In the fifteenth over of a tense run chase under lights, the leg-spinner",
        "Deep into the death overs with a packed off-side ring, the express bowler",
    ]
    deliveries = [
        "steamed in from the pavilion end with a reversing old ball and fired a searing yorker at ninety miles an hour.",
        "hit a nagging fourth-stump channel and shaped a late outswinger past the outside edge of the bat.",
        "disguised a sharp googly in flight to beat the left-hander on the inside edge and hit middle stump.",
        "executed wide yorkers outside the tramline, conceding only singles while building pressure on the crease.",
        "dug in a sharp bouncer before following up with a slower off-cutter into the pitch.",
    ]
    batting = [
        "The set batter backed away to the leg side and carved a fierce square cut past backward point to the boundary rope for four.",
        "The captain rotated the strike with nimble singles and quick twos before launching a towering pull shot over deep midwicket for six.",
        "The opener cleared his front leg and struck an effortless cover drive cleanly through the infield gap.",
        "The finisher stayed calm at the crease, piercing gaps behind point with late cuts and lofting an inside-out drive over extra cover.",
        "A delicate scoop over fine leg brought up a match-winning fifty off thirty-two deliveries.",
    ]
    closings = [
        "A brilliant diving catch on the boundary rope turned the tide in a nail-biting finish.",
        "Sprinting back for the second run off the last ball, the batters beat the throw from the deep by inches.",
        "The fielding side defended a modest total by five runs after two crucial wickets in the penultimate over.",
        "Clinical death bowling and fearless strokeplay produced an unforgettable contest under lights.",
        "The crowd rose in unison as the winning boundary raced across the outfield turf.",
    ]

    articles: list[str] = []
    seen: set[str] = set()
    while len(articles) < 34:
        article = (
            f"{rng.choice(titles)}: FINAL-OVER DRAMA\n"
            f"{rng.choice(openers)} {rng.choice(deliveries)} {rng.choice(batting)}\n"
            f"{rng.choice(closings)}"
        )
        if article not in seen:
            seen.add(article)
            articles.append(article)
    return articles


def _generate_baseball_articles(rng: random.Random) -> list[str]:
    titles = [
        "WORLD SERIES PLAY-BY-PLAY",
        "BALLPARK NIGHT REPORT",
        "PENNANT RACE COMMENTARY",
        "INNING-BY-INNING RECAP",
    ]
    openers = [
        "Bottom of the ninth inning, two outs, runners on first and second, and a full count on the cleanup hitter.",
        "Top of the eighth inning with the bases loaded and one out, and the reliever enters needing a ground ball.",
        "Leading off the sixth inning in a scoreless pitching duel,",
        "With two outs in the bottom of the eighth and the tying run on second base,",
        "Late in the ninth inning with the closer staring in for the sign,",
    ]
    pitches = [
        "The closer shook off a slider and fired a ninety-eight-mile-per-hour four-seam fastball on the inner half.",
        "The right-hander pounded a heavy sinker onto the inside corner, jamming the batter into a sharp grounder.",
        "After fouling off three tough two-strike breaking balls, the hitter got a hanging changeup on the outer half.",
        "The setup pitcher dropped a sharp downward curveball out of the strike zone to earn a crucial swinging strikeout.",
        "The starter painted corners on the outside edge through seven innings with twelve strikeouts.",
    ]
    outcomes = [
        "The slugger turned on it with thunderous bat speed, sending a towering fly ball over the right-field wall for a walk-off home run.",
        "The shortstop ranged left, flipped cleanly to second base, and completed a sparkling 6-4-3 double play to escape the jam.",
        "A sharp line-drive single into center field brought the runner racing around third to slide safely across home plate.",
        "He lashed a ringing double down the left-field line as the go-ahead run scored standing up.",
        "The speedster broke on the pitcher's first move and slid headfirst under the tag to steal second base cleanly.",
    ]
    closings = [
        "The bullpen slammed the door in the ninth inning with three consecutive strikeouts.",
        "The stadium waved towels in celebration after an unforgettable late-inning rally.",
        "The right fielder leaped at the warning track, watching the ball sail into the roaring crowd.",
        "Sharp defense up the middle and fearless bullpen execution secured the series victory.",
        "Both lineups battled every pitch in a tense postseason atmosphere.",
    ]

    articles: list[str] = []
    seen: set[str] = set()
    while len(articles) < 34:
        article = (
            f"{rng.choice(titles)}: LATE-INNING RALLY\n"
            f"{rng.choice(openers)} {rng.choice(pitches)} {rng.choice(outcomes)}\n"
            f"{rng.choice(closings)}"
        )
        if article not in seen:
            seen.add(article)
            articles.append(article)
    return articles


def _generate_tennis_articles(rng: random.Random) -> list[str]:
    titles = [
        "GRAND SLAM PLAY-BY-PLAY",
        "CENTER COURT REPORT",
        "CHAMPIONSHIP MATCH COMMENTARY",
        "TIEBREAK TENSION RECAP",
    ]
    openers = [
        "Serving for the championship at five-four in the deciding fifth set,",
        "Fourth-set tiebreak locked at six-all as both players refuse to blink on serve.",
        "Serving at five-all in the third-set tiebreak under bright stadium lights,",
        "At deuce deep into the fourth set after three hours on court,",
        "Facing double break point late in the deciding set,",
    ]
    serves = [
        "the top seed pounded a 132-mile-per-hour ace down the T before charging the net.",
        "the challenger stepped inside the baseline to crush a flat backhand return deep at his opponent's shoelaces.",
        "the defending champion landed a biting kick serve out wide to pull the returner off the court.",
        "he mixed pace brilliantly, sliding a wide slice serve into the ad court on big points.",
        "a kicking second serve forced a short return into the middle of the court.",
    ]
    rallies = [
        "A grueling twenty-eight-shot rally unfolded across every corner until a breathtaking backhand passing shot kissed the sideline.",
        "Following the serve straight to the net, he absorbed a low pass with a cushioned drop volley that died inches over the tape.",
        "Absorbing heavy topspin groundstrokes behind the baseline, he disguised a delicate drop shot that bounced twice before the net.",
        "He put away a weak half-volley with a crisp overhead smash into the open corner of the court.",
        "A fearless inside-out forehand winner clipped the chalk to clinch the game and raise a clenched fist.",
    ]
    closings = [
        "High first-serve percentages and aggressive court positioning kept pressure squarely on the returner.",
        "On championship point, a clean crosscourt volley sealed the title after four grueling hours.",
        "The center court crowd rose in applause for an extraordinary display of poise under pressure.",
        "Clinical net play and relentless baseline defense decided a five-set classic.",
        "One unreturnable ace down the center line closed out the set in emphatic style.",
    ]

    articles: list[str] = []
    seen: set[str] = set()
    while len(articles) < 34:
        article = (
            f"{rng.choice(titles)}: DECIDING-SET DRAMA\n"
            f"{rng.choice(openers)} {rng.choice(serves)} {rng.choice(rallies)}\n"
            f"{rng.choice(closings)}"
        )
        if article not in seen:
            seen.add(article)
            articles.append(article)
    return articles


def _build_sports_articles() -> list[str]:
    rng = random.Random(42)
    generated = (
        _generate_football_articles(rng)
        + _generate_basketball_articles(rng)
        + _generate_cricket_articles(rng)
        + _generate_baseball_articles(rng)
        + _generate_tennis_articles(rng)
    )
    rng.shuffle(generated)

    combined: list[str] = []
    seen: set[str] = set()
    for article in _BASE_ARTICLES + generated:
        if article not in seen:
            seen.add(article)
            combined.append(article)
    return combined


SPORTS_ARTICLES: list[str] = _build_sports_articles()
