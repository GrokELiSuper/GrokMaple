"""A collection of all commands that a Kanna can use to interact with the game in Maple nunu world."""

from src.common import config, settings, utils
import time
import math
from src.routine.components import Command
from src.common.vkeys import press, key_down, key_up

# List of key mappings
class Key:
    # Movement
    JUMP = 'c'           # Updated to match your binding
    TELEPORT = 'space'   # Updated to match your binding
    DOUBLE_JUMP = '>'    # Added for your double jump

    # Buffs/Damage Boosts
    NIKA_MODE = '1'      # Increases damage by 60%
    DAMAGE_BOOST = '`'   # Further increases damage with Nika mode

    # Skills
    MAIN_ATTACK = 'z'    # Main hitting skill
    MODE_CHANGE = '2'    # Toggles between default and barrier mode
    BARRIER_ATTACK = ',' # Barrier-breaking attack in barrier mode
    SPEED_BOOST = 'm'    # Increases speed of barrier attack

#########################
#       Commands        #
#########################
def step(direction, target):
    """
    Performs one movement step in the given DIRECTION towards TARGET.
    Should not press any arrow keys, as those are handled by Auto Maple.
    """
    num_presses = 2
    if direction == 'up' or direction == 'down':
        num_presses = 1
    if config.stage_fright and direction != 'up' and utils.bernoulli(0.75):
        time.sleep(utils.rand_float(0.1, 0.3))
    d_y = target[1] - config.player_pos[1]
    if abs(d_y) > settings.move_tolerance * 1.5:
        if direction == 'down':
            press(Key.JUMP, 3)
        elif direction == 'up':
            press(Key.JUMP, 1)
    press(Key.TELEPORT, num_presses)

class Adjust(Command):
    """Fine-tunes player position using small movements."""
    def __init__(self, x, y, max_steps=5):
        super().__init__(locals())
        self.target = (float(x), float(y))
        self.max_steps = settings.validate_nonnegative_int(max_steps)

    def main(self):
        counter = self.max_steps
        toggle = True
        error = utils.distance(config.player_pos, self.target)
        while config.enabled and counter > 0 and error > settings.adjust_tolerance:
            if toggle:
                d_x = self.target[0] - config.player_pos[0]
                threshold = settings.adjust_tolerance / math.sqrt(2)
                if abs(d_x) > threshold:
                    walk_counter = 0
                    if d_x < 0:
                        key_down('left')
                        while config.enabled and d_x < -1 * threshold and walk_counter < 60:
                            time.sleep(0.05)
                            walk_counter += 1
                            d_x = self.target[0] - config.player_pos[0]
                        key_up('left')
                    else:
                        key_down('right')
                        while config.enabled and d_x > threshold and walk_counter < 60:
                            time.sleep(0.05)
                            walk_counter += 1
                            d_x = self.target[0] - config.player_pos[0]
                        key_up('right')
                    counter -= 1
            else:
                d_y = self.target[1] - config.player_pos[1]
                if abs(d_y) > settings.adjust_tolerance / math.sqrt(2):
                    if d_y < 0:
                        Teleport('up').main()
                    else:
                        key_down('down')
                        time.sleep(0.05)
                        press(Key.JUMP, 3, down_time=0.1)
                        key_up('down')
                        time.sleep(0.05)
                    counter -= 1
            error = utils.distance(config.player_pos, self.target)
            toggle = not toggle

class Buff(Command):
    """Manages damage-boosting modes and buffs (Nika mode and additional boost)."""
    def __init__(self):
        super().__init__(locals())
        self.nika_time = 0
        self.damage_boost_active = False
        self.damage_multiplier = 1.0

    def main(self):
        now = time.time()
        # Nika mode: 60% damage increase
        if self.nika_time == 0 or now - self.nika_time > 120:  # Assumed 120s cooldown
            if key_down(Key.NIKA_MODE):
                self.damage_multiplier = 1.6  # 60% increase
                self.nika_time = now
                print("Nika mode activated: 60% damage increase")
        # Additional damage boost with Nika mode
        if key_down(Key.DAMAGE_BOOST) and self.damage_multiplier > 1.0:
            self.damage_multiplier *= 1.2  # Additional 20% increase
            self.damage_boost_active = True
            print("Additional damage boost activated: 20% extra")
        else:
            self.damage_boost_active = False

class Teleport(Command):
    """
    Teleports in a given direction, jumping if specified. Adds the player's position
    to the current Layout if necessary.
    """
    def __init__(self, direction, jump='False'):
        super().__init__(locals())
        self.direction = settings.validate_arrows(direction)
        self.jump = settings.validate_boolean(jump)

    def main(self):
        num_presses = 3
        time.sleep(0.05)
        if self.direction in ['up', 'down']:
            num_presses = 2
        if self.direction != 'up':
            key_down(self.direction)
            time.sleep(0.05)
        if self.jump:
            if self.direction == 'down':
                press(Key.JUMP, 3, down_time=0.1)
            else:
                press(Key.JUMP, 1)
        if self.direction == 'up':
            key_down(self.direction)
            time.sleep(0.05)
        press(Key.TELEPORT, num_presses)
        key_up(self.direction)
        if settings.record_layout:
            config.layout.add(*config.player_pos)

class DoubleJump(Command):
    """Performs a double jump."""
    def main(self):
        press(Key.JUMP, 1, down_time=0.1, up_time=0.05)
        press(Key.DOUBLE_JUMP, 1, down_time=0.1, up_time=0.05)

class MainAttack(Command):
    """Attacks using the main hitting skill in a given direction."""
    def __init__(self, direction, attacks=2, repetitions=1):
        super().__init__(locals())
        self.direction = settings.validate_horizontal_arrows(direction)
        self.attacks = int(attacks)
        self.repetitions = int(repetitions)
        self.mode = 'default'  # Tracks mode (default or barrier)

    def main(self):
        # Check mode by simulating key press detection
        if key_down(Key.MODE_CHANGE):
            self.mode = 'barrier' if self.mode == 'default' else 'default'
            time.sleep(0.1)  # Small delay to prevent rapid toggling
            print(f"Mode changed to: {self.mode}")

        if self.mode == 'default':
            time.sleep(0.05)
            key_down(self.direction)
            time.sleep(0.05)
            if config.stage_fright and utils.bernoulli(0.7):
                time.sleep(utils.rand_float(0.1, 0.3))
            for _ in range(self.repetitions):
                # Apply damage multiplier from Buff
                damage = 100 * config.bot.command_book['buff'].damage_multiplier  # Base damage 100
                print(f"Main attack damage: {damage}")
                press(Key.MAIN_ATTACK, self.attacks, up_time=0.05)
            key_up(self.direction)
            if self.attacks > 2:
                time.sleep(0.3)
            else:
                time.sleep(0.2)

class BarrierAttack(Command):
    """Attacks using the barrier-breaking skill in a given direction with speed boost."""
    def __init__(self, direction, attacks=2, repetitions=1):
        super().__init__(locals())
        self.direction = settings.validate_horizontal_arrows(direction)
        self.attacks = int(attacks)
        self.repetitions = int(repetitions)
        self.mode = 'default'
        self.speed_multiplier = 1.0

    def main(self):
        # Check mode
        if key_down(Key.MODE_CHANGE):
            self.mode = 'barrier' if self.mode == 'default' else 'default'
            time.sleep(0.1)
            print(f"Mode changed to: {self.mode}")

        # Speed boost
        if key_down(Key.SPEED_BOOST):
            self.speed_multiplier = 1.5  # 50% speed increase
            print("Barrier attack speed increased")

        if self.mode == 'barrier':
            time.sleep(0.05)
            key_down(self.direction)
            time.sleep(0.05)
            if config.stage_fright and utils.bernoulli(0.7):
                time.sleep(utils.rand_float(0.1, 0.3))
            for _ in range(self.repetitions):
                # Apply damage multiplier from Buff
                damage = 120 * config.bot.command_book['buff'].damage_multiplier  # Base damage 120 (higher for barrier)
                print(f"Barrier attack damage: {damage}")
                press(Key.BARRIER_ATTACK, self.attacks, up_time=0.05 / self.speed_multiplier)
            key_up(self.direction)
            if self.attacks > 2:
                time.sleep(0.3 / self.speed_multiplier)
            else:
                time.sleep(0.2 / self.speed_multiplier)