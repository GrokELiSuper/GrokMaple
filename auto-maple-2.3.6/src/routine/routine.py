"""A collection of classes that represent routines and their components."""

import time
import os
from src.common import config, utils
from src.routine.components import Command, Point, Wait, Walk, Fall, Move, Label, Jump, Setting, SYMBOLS
from src.common.vkeys import key_down, key_up

class Routine:
    def __init__(self):
        self.sequence = []
        self.index = 0
        self.commands = {
            'wait': Wait,
            'walk': Walk,
            'fall': Fall,
            'move': Move,
            'point': Point  # Register the Point command
        }
        self.prev_dir = 'left'
        self.labels = {}
        self.dirty = False  # Track changes to the routine
        self.path = None

    def add(self, *args):
        """Adds a new Point to this Routine's sequence."""
        for p in args:
            if isinstance(p, Point):
                self.sequence.append(p)
                self.dirty = True

    def append(self, component):
        """Appends a component to the routine sequence."""
        if isinstance(component, (Point, Label, Jump, Setting)):
            self.sequence.append(component)
            if isinstance(component, Label):
                self.labels[component.label] = component
            self.dirty = True

    def clear(self):
        """Removes all Points from this Routine."""
        self.sequence = []
        self.index = 0
        for label in self.labels.values():
            label.links.clear()
        self.labels.clear()
        self.dirty = True

    def step(self):
        """Increments this Routine's index by 1, looping back to 0 if necessary."""
        self.index = (self.index + 1) % len(self.sequence)
        self.dirty = True

    def reset(self):
        """Resets this Routine's index to 0."""
        self.index = 0
        self.dirty = True

    def load(self, file_path):
        """Loads a routine from a file."""
        print(f"\n[~] Loading routine '{file_path}':")
        self.path = file_path
        self.clear()
        config.routine = self  # Update the global routine reference

        with open(file_path) as f:
            lines = f.readlines()
            for i, raw_line in enumerate(lines):
                line = raw_line.strip()
                if line and not line.startswith('#'):
                    try:
                        args = line.split(',')
                        args = [arg.strip() for arg in args]
                        if not args:
                            raise ValueError("Empty line")
                        symbol = args[0]
                        if symbol not in SYMBOLS:
                            print(f" !  Line {i + 1}: Command '{symbol}' does not exist.")
                            continue
                        component_class = SYMBOLS[symbol]
                        # Parse arguments for the component
                        kwargs = {}
                        positional_args = []
                        for arg in args[1:]:
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                kwargs[key.strip()] = value.strip()
                            else:
                                positional_args.append(arg)
                        # Create the component
                        component = component_class(*positional_args, **kwargs)
                        self.append(component)
                    except Exception as e:
                        print(f" !  Line {i + 1}: {str(e)}")
                        continue

        # Bind jumps to labels
        for i, component in enumerate(self.sequence):
            if isinstance(component, Label):
                component.set_index(i)
            elif isinstance(component, Jump):
                if not component.bind():
                    print(f" !  Jump at line {i + 1}: Label '{component.label}' does not exist.")

        print(f" ~  Finished loading routine '{os.path.basename(file_path)}'.")

    def save(self, file_path=None):
        """Saves the routine to a file."""
        if file_path is None:
            file_path = self.path
        if file_path is None:
            raise ValueError("No file path specified for saving routine.")
        with open(file_path, 'w') as f:
            for component in self.sequence:
                f.write(component.encode() + '\n')
        self.dirty = False

    def get_all_components(self):
        """Returns a dictionary of all available components for the GUI."""
        components = {}
        # Add routine components (Point, Label, Jump, Setting)
        for symbol, component_class in SYMBOLS.items():
            components[symbol] = component_class
        # Add command book components
        for name, command in config.bot.command_book.items():
            components[name] = command
        return components

    def get_component(self, index):
        """Returns the component at the specified index."""
        if 0 <= index < len(self.sequence):
            return self.sequence[index]
        return None

    def insert(self, index, component):
        """Inserts a component at the specified index."""
        if isinstance(component, (Point, Label, Jump, Setting)):
            self.sequence.insert(index, component)
            if isinstance(component, Label):
                self.labels[component.label] = component
            self.dirty = True

    def remove(self, index):
        """Removes the component at the specified index."""
        if 0 <= index < len(self.sequence):
            component = self.sequence.pop(index)
            if isinstance(component, Label):
                self.labels.pop(component.label, None)
            self.dirty = True

class RoutineComponent:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.start_time = 0

    def execute(self):
        """Executes this RoutineComponent."""
        self.start_time = time.time()
        self._execute()

    def _execute(self):
        pass

    def duration(self):
        """Returns how long this RoutineComponent has been executing for."""
        return time.time() - self.start_time

class Wait(RoutineComponent):
    """Waits for a specified amount of time."""

    def __init__(self, t, **kwargs):
        super().__init__(kwargs)
        self.t = float(t)

    def _execute(self):
        time.sleep(self.t)

class Walk(RoutineComponent):
    """Walks in a specified direction for a specified amount of time."""

    def __init__(self, direction, t, **kwargs):
        super().__init__(kwargs)
        self.direction = direction.lower()
        self.t = float(t)

    def _execute(self):
        key_down(self.direction)
        time.sleep(self.t)
        key_up(self.direction)

class Fall(RoutineComponent):
    """Falls for a specified amount of time."""

    def __init__(self, t, **kwargs):
        super().__init__(kwargs)
        self.t = float(t)

    def _execute(self):
        key_down('down')
        time.sleep(self.t)
        key_up('down')

class Move(RoutineComponent):
    """Moves to a specified location on the minimap."""

    def __init__(self, x, y, **kwargs):
        super().__init__(kwargs)
        self.target = (float(x), float(y))
        self.direction = 'left'

    def _execute(self):
        # Adjust target position based on map offset (dynamic minimap)
        map_offset = config.capture.map_offset
        adjusted_target = (
            self.target[0] + map_offset[0],
            self.target[1] + map_offset[1]
        )
        config.routine.commands['step'](self.direction, adjusted_target)

class Point(RoutineComponent):
    """Represents a single point in a Routine."""

    def __init__(self, x, y, command, **kwargs):
        super().__init__(kwargs)
        self.location = (float(x), float(y))
        self.command = command.lower()

    def _execute(self):
        if self.command in config.bot.command_book:
            command = config.bot.command_book[self.command]
            command(self.location, **self.kwargs).execute()
        elif self.command in config.routine.commands:
            command = config.routine.commands[self.command]
            command(self.location, **self.kwargs).execute()
        else:
            print(f"\n[!] Unknown command '{self.command}'")