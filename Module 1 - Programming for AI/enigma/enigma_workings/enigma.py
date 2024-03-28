class EnigmaConfig:
    enigma_data = {
        'Alphabet': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z'],
        'Notches': [('I', 'Q'), ('II', 'E'), ('III', 'V'), ('IV', 'J'), ('V', 'Z'), ('Beta', 'None'),
                    ('Gamma', 'None')],
        'Beta': ['L', 'E', 'Y', 'J', 'V', 'C', 'N', 'I', 'X', 'W', 'P', 'B', 'Q', 'M', 'D', 'R', 'T', 'A', 'K', 'Z',
                 'G', 'F', 'U', 'H', 'O', 'S'],
        'Gamma': ['F', 'S', 'O', 'K', 'A', 'N', 'U', 'E', 'R', 'H', 'M', 'B', 'T', 'I', 'Y', 'C', 'W', 'L', 'Q', 'P',
                  'Z', 'X', 'V', 'G', 'J', 'D'],
        'I': ['E', 'K', 'M', 'F', 'L', 'G', 'D', 'Q', 'V', 'Z', 'N', 'T', 'O', 'W', 'Y', 'H', 'X', 'U', 'S', 'P', 'A',
              'I', 'B', 'R', 'C', 'J'],
        'II': ['A', 'J', 'D', 'K', 'S', 'I', 'R', 'U', 'X', 'B', 'L', 'H', 'W', 'T', 'M', 'C', 'Q', 'G', 'Z', 'N', 'P',
               'Y', 'F', 'V', 'O', 'E'],
        'III': ['B', 'D', 'F', 'H', 'J', 'L', 'C', 'P', 'R', 'T', 'X', 'V', 'Z', 'N', 'Y', 'E', 'I', 'W', 'G', 'A', 'K',
                'M', 'U', 'S', 'Q', 'O'],
        'IV': ['E', 'S', 'O', 'V', 'P', 'Z', 'J', 'A', 'Y', 'Q', 'U', 'I', 'R', 'H', 'X', 'L', 'N', 'F', 'T', 'G', 'K',
               'D', 'C', 'M', 'W', 'B'],
        'V': ['V', 'Z', 'B', 'R', 'G', 'I', 'T', 'Y', 'U', 'P', 'S', 'D', 'N', 'H', 'L', 'X', 'A', 'W', 'M', 'J', 'Q',
              'O', 'F', 'E', 'C', 'K'],
        'A': ['E', 'J', 'M', 'Z', 'A', 'L', 'Y', 'X', 'V', 'B', 'W', 'F', 'C', 'R', 'Q', 'U', 'O', 'N', 'T', 'S', 'P',
              'I', 'K', 'H', 'G', 'D'],
        'B': ['Y', 'R', 'U', 'H', 'Q', 'S', 'L', 'D', 'P', 'X', 'N', 'G', 'O', 'K', 'M', 'I', 'E', 'B', 'F', 'Z', 'C',
              'W', 'V', 'J', 'A', 'T'],
        'C': ['F', 'V', 'P', 'J', 'I', 'A', 'O', 'Y', 'E', 'D', 'R', 'Z', 'X', 'W', 'G', 'C', 'T', 'K', 'U', 'Q', 'S',
              'B', 'N', 'M', 'H', 'L']}


class PlugLead:
    def __init__(self, character_map):
        self.character_map = character_map
        self.character_1 = list(character_map)[0]
        self.character_2 = list(character_map)[1]

    def encode(self, character):
        # Returns letter pairing of lead
        if character == self.character_1:
            return self.character_2
        elif character == self.character_2:
            return self.character_1
        return character


class Plugboard:
    def __init__(self):
        self.leads = []

    def add(self, plug_lead):
        self.leads.append(plug_lead)

    def encode(self, character):
        # Loops through PlugLead objects appended to the Plugboard and returns the encoding
        # If letter present in a PlugLead, return encoded character, if not, returns the same character
        for lead in self.leads:
            encoded_character = lead.encode(character)
            if character != encoded_character:
                return encoded_character
        return character

    def delete(self):
        self.leads = self.leads[:-2]
<<<<<<< HEAD:Module 1 - Principles of Programming for AI/enigma/enigma_workings/enigma.py


=======
        

 
 
>>>>>>> master:Module 1 - Principles of Programming for AI/enigma/enigma.py
class Rotor_Adjuster():
    def __init__(self, ring_settings):
        self.ring_settings = ring_settings

    def rotor_offsetter(self, position_offset):
<<<<<<< HEAD:Module 1 - Principles of Programming for AI/enigma/enigma_workings/enigma.py
        rotor_order = []
        # Loop through each character in the alphabet
        for i in range(26):
            # Calculate the new index using modular arithmetic to handle wrapping
            offset_index = (i + position_offset) % 26

            # Append the character at the offset index to the rotor order after accessing the alphabet from Enigma Config
            rotor_order.append(EnigmaConfig.enigma_data['Alphabet'][offset_index])
        return rotor_order

    def rotor_transition(self, rotor_transition_offset, char):
        # Find the index of the character in the alphabet.
        char_idx = EnigmaConfig.enigma_data['Alphabet'].index(char) + rotor_transition_offset

        # Apply the offset and use modular arithmetic to wrap around if needed
        char = EnigmaConfig.enigma_data['Alphabet'][char_idx % 26]
        return char

    def ring_offsetter(self, position_settings, rev):
        ring_settings = self.ring_settings

        # If 'rev' is True, reverse the ring settings. This is only used by the Rotor Tuner logic as I use a reversed list there
        # to avoid the index going out of bounds when I add a fourth rotor
        if rev:
            ring_settings = [26 - setting + 2 for setting in ring_settings]

        # Create a list of settings tuples combining ring and position settings.
        settings_list = [(ring_settings[i] - 1, position) for i, position in enumerate(position_settings)]
        offset_position_settings = []
        for ring_offset, position in settings_list:
            if position != 'None':

                # Calculate the character's index after applying the ring offset
                character_idx = EnigmaConfig.enigma_data['Alphabet'].index(position)
                offset_idx = character_idx - ring_offset

                # Ensure the offset index wraps around correctly
                offset_idx %= 26
=======
        alphabet = EnigmaConfig.enigma_data['Alphabet']
        rotor_order = [alphabet[(i + position_offset) % 26] for i in range(26)]
        return rotor_order

    def rotor_transition(self, rotor_transition_offset, char):
        alphabet_idx = EnigmaConfig.enigma_data['Alphabet'].index(char)
        transitioned_char = EnigmaConfig.enigma_data['Alphabet'][(alphabet_idx + rotor_transition_offset) % 26]
        return transitioned_char

    def ring_offsetter(self, position_settings, reverse=False):
        offset_position_settings = []
        settings_list = [(self.ring_settings[i] - 1, position) for i, position in enumerate(position_settings)]
        if reverse:
            self.ring_settings = [26 - setting + 2 for setting in self.ring_settings]
        for setting, position in settings_list:
            if position != 'None':
                character_idx = EnigmaConfig.enigma_data['Alphabet'].index(position)
                offset_idx = (character_idx - setting) % 26
>>>>>>> master:Module 1 - Principles of Programming for AI/enigma/enigma.py
                offset_character = EnigmaConfig.enigma_data['Alphabet'][offset_idx]
                offset_position_settings.append(offset_character)
            else:
                offset_position_settings.append(position)
<<<<<<< HEAD:Module 1 - Principles of Programming for AI/enigma/enigma_workings/enigma.py

        # Return the ring adjusted rotor positiions
=======
>>>>>>> master:Module 1 - Principles of Programming for AI/enigma/enigma.py
        return offset_position_settings

    def set_ring_settings(self, ring_settings):
        self.ring_settings = ring_settings
<<<<<<< HEAD:Module 1 - Principles of Programming for AI/enigma/enigma_workings/enigma.py
=======


>>>>>>> master:Module 1 - Principles of Programming for AI/enigma/enigma.py


class Notch_Adjuster():
    def __init__(self, ring_settings, rotor_engine, rotor_adjuster):
        self.ring_settings = ring_settings
        self.rotor_engine = rotor_engine
        self.rotor_adjuster = rotor_adjuster

    def adjust(self, rotor_start_positions):
<<<<<<< HEAD:Module 1 - Principles of Programming for AI/enigma/enigma_workings/enigma.py
        # Extract rotor start positions
        rotor_positions = [pos[0] for pos in rotor_start_positions]

        # Fetch rotor types and match them with notches
        position_notches = []
        unadjusted_notches = []
        for rotor_type in self.rotor_engine.get_rotors():
=======
        rotor_positions = [position[0] for position in rotor_start_positions]
        position_notches = []
        unadjusted_notches = []
        for rotor in self.rotor_engine.get_rotors():
>>>>>>> master:Module 1 - Principles of Programming for AI/enigma/enigma.py
            for notch in EnigmaConfig.enigma_data['Notches']:
                if notch[0] == rotor.get_rotor_type():
                    position_notches.append(notch[0])
                    unadjusted_notches.append(notch[1])

        # Adjust notches
        adjusted_notches = self.rotor_adjuster.ring_offsetter(unadjusted_notches, False)
<<<<<<< HEAD:Module 1 - Principles of Programming for AI/enigma/enigma_workings/enigma.py

        # Combine rotor types and adjusted notches
        final_position_notches = list(zip(position_notches, adjusted_notches))

        return final_position_notches, rotor_positions

    def set_ring_settings(self, ring_settings):
        self.ring_settings = ring_settings
=======
        final_position_notches = list(zip(position_notches, adjusted_notches))
        return final_position_notches, rotor_positions
>>>>>>> master:Module 1 - Principles of Programming for AI/enigma/enigma.py

    def set_ring_settings(self, ring_settings):
        self.ring_settings = ring_settings
<<<<<<< HEAD:Module 1 - Principles of Programming for AI/enigma/enigma_workings/enigma.py
        self.rotor_adjuster = rotor_adjuster

    def turn(self, notch_data, rotor_positions):
        alphabet = EnigmaConfig.enigma_data['Alphabet']

        # Switch to handle the double notch
        switch = False

        # Reverse to handle out of bounds indexing due to rotor[0] being the left most rotor and I need to start with the right most rotor
        rotor_positions.reverse()
        notch_data.reverse()

        # Get current rotor position indicies
        rotor_indices = [alphabet.index(pos) for pos in rotor_positions[:3]]
        rotor_1_idx, rotor_2_idx, rotor_3_idx = rotor_indices
        rotor_1_char = alphabet[rotor_1_idx % 26]
        rotor_2_char = rotor_positions[1]
        rotor_3_char = rotor_positions[2]

        # Handle the notch turning the middle and left-most rotors and handles the double notch
        if rotor_2_char == notch_data[1][1] and notch_data[1][0] not in ('Beta', 'Gamma') and not switch:
            switch = True
            rotor_2_idx += 1
            rotor_2_char = alphabet[rotor_2_idx % 26]
            rotor_3_idx += 1
            rotor_3_char = alphabet[rotor_3_idx % 26]
        if rotor_1_char == notch_data[0][1] and notch_data[0][0] not in ('Beta', 'Gamma') and not switch:
            rotor_2_idx += 1
            rotor_2_char = alphabet[rotor_2_idx % 26]

        # Increment the right most rotor every key press
        rotor_1_idx += 1
        rotor_1_char = alphabet[rotor_1_idx % 26]

        # 4th rotor logic
        if len(rotor_positions) == 4:
            turned_positions_char = [rotor_1_char, rotor_2_char, rotor_3_char, rotor_positions[3]]
            turned_position_idx = [rotor_1_idx, rotor_2_idx, rotor_3_idx, alphabet.index(rotor_positions[3])]
        else:
            turned_positions_char = [rotor_1_char, rotor_2_char, rotor_3_char]
            turned_position_idx = [rotor_1_idx, rotor_2_idx, rotor_3_idx]

        # Reverse the reverse logic
        turned_positions_char.reverse()
        turned_position_idx.reverse()

        # Convert the output to respect the rotor start positions and ring settings as the notch letter accounts for both of these.
        turned_position_rotors = [self.rotor_adjuster.rotor_offsetter(i) for i in turned_position_idx]
        turned_base_positions = self.rotor_adjuster.ring_offsetter(turned_positions_char, True)

        return turned_position_rotors, turned_base_positions

    def set_ring_settings(self, ring_settings):
        self.ring_settings = ring_settings
=======

>>>>>>> master:Module 1 - Principles of Programming for AI/enigma/enigma.py


class Rotor_Engine:
    def __init__(self, rotor_adjuster, reflector):
        self.rotors = []
        self.rotor_adjuster = rotor_adjuster
        self.reflector = reflector

    def encode_controller(self, character_in, rotor_positions):

        # Reverses the positions so out of bounds indexing does not occur with a 4th rotor.
        rotor_positions.reverse()

        # Calculates the rotor offsets so the encoder function knows how to translate the character before encoding the character through the internal wirings
        rotor_offsets = []
        for i in rotor_positions:
            rotor_offsets.append(EnigmaConfig.enigma_data['Alphabet'].index(i[0]))

        # Controlls the encoding through the rotors from right to left, hitting the reflector and back through.
        # Passes in the base rotor positions, the offsets, the character and the identifier to tell the encoder function which rotor this is.
        encoded_char = self.encoder(rotor_positions, rotor_offsets, 0, None, character_in)
        encoded_char = self.encoder(rotor_positions, rotor_offsets, 1, 'Forward', encoded_char)
        encoded_char = self.encoder(rotor_positions, rotor_offsets, 2, 'Forward', encoded_char)
        if self.get_rotors_count() == 4:
            encoded_char = self.encoder(rotor_positions, rotor_offsets, 3, 'Forward', encoded_char)
        encoded_char = self.reflector.encode(rotor_offsets, encoded_char)
        if self.get_rotors_count() == 4:
            encoded_char = self.encoder(rotor_positions, rotor_offsets, 3, 'Reflector', encoded_char)
        if self.get_rotors_count() == 4:
            encoded_char = self.encoder(rotor_positions, rotor_offsets, 2, 'Backward', encoded_char)
        else:
            encoded_char = self.encoder(rotor_positions, rotor_offsets, 2, 'Reflector', encoded_char)
        encoded_char = self.encoder(rotor_positions, rotor_offsets, 1, 'Backward', encoded_char)
        encoded_char = self.encoder(rotor_positions, rotor_offsets, 0, 'Backward', encoded_char)

        # After running the character through the rotor engine, it then translates to the rotor housing by comaring to the unoffsetted alphabet and then passes this back to the plugboard.
        output_char = EnigmaConfig.enigma_data['Alphabet'][(EnigmaConfig.enigma_data['Alphabet'].index(encoded_char) -
                                                            EnigmaConfig.enigma_data['Alphabet'].index(
                                                                rotor_positions[0][0])) % 26]
        return output_char

    def encoder(self, rotor_positions, rotor_offsets, idx, rotor_transition_offset, input_char):
        rotor_position = rotor_positions[idx]

        # Right most rotor always faces a static rotor housing, so no offsetting needed before encoding the character via the internal wirings
        if rotor_transition_offset == None:
            rotor_char_idx = EnigmaConfig.enigma_data['Alphabet'].index(input_char)
            rotor_char = rotor_position[rotor_char_idx]

        # Rotor translation for hitting the reflector
        elif rotor_transition_offset == 'Reflector':
            rotor_char = self.rotor_adjuster.rotor_transition(rotor_offsets[idx], input_char)

        # Rotor translation for right to left rotors
        elif rotor_transition_offset == 'Forward':
            rotor_char = self.rotor_adjuster.rotor_transition(rotor_offsets[idx] - rotor_offsets[idx - 1], input_char)
        else:
            # Rotor translation for left to right rotors
            if idx == 3:
                rotor_char = self.rotor_adjuster.rotor_transition(rotor_offsets[idx], input_char)
            else:
                rotor_char = self.rotor_adjuster.rotor_transition(rotor_offsets[idx] - rotor_offsets[idx + 1],
                                                                  input_char)

        # Encodes the character via the rotors internal wirings
        if rotor_transition_offset == 'Forward' or rotor_transition_offset == None:
            return self.rotors[::-1][idx].encode_right_to_left(rotor_char)
        else:
            return self.rotors[::-1][idx].encode_left_to_right(rotor_char)

    def add_rotor(self, rotor):
        self.rotors.append(rotor)

    def get_rotors_count(self):
        return len(self.rotors)

    def get_rotors(self):
        return self.rotors

<<<<<<< HEAD:Module 1 - Principles of Programming for AI/enigma/enigma_workings/enigma.py
    def add_rotor(self, rotor):
        self.rotors.append(rotor)
=======
    def reset_rotors(self):
        self.rotors = []
>>>>>>> master:Module 1 - Principles of Programming for AI/enigma/enigma.py

    def set_reflector(self, reflector):
        self.reflector = reflector

<<<<<<< HEAD:Module 1 - Principles of Programming for AI/enigma/enigma_workings/enigma.py
    def reset_rotors(self):
        self.rotors = []
=======
>>>>>>> master:Module 1 - Principles of Programming for AI/enigma/enigma.py


class Rotor:
    def __init__(self, rotor_type):
        self.rotor_type = rotor_type

    def encode_right_to_left(self, character):
        # Take in a character and encodes via the rotors internal wiring
        char_idx = EnigmaConfig.enigma_data['Alphabet'].index(character)
        return EnigmaConfig.enigma_data[self.rotor_type][char_idx]

    def encode_left_to_right(self, character):
        # Take in the encoded character from the rotors internal wiring and outputs the respective alphabetic letter
        char_idx = EnigmaConfig.enigma_data[self.rotor_type].index(character)
        return EnigmaConfig.enigma_data['Alphabet'][char_idx]

    def get_rotor_type(self):
        return self.rotor_type


class Reflector:
    def __init__(self, reflector_type, rotor_adjuster):
        self.reflector_type = reflector_type
        self.rotor_count = 0
        self.rotor_adjuster = rotor_adjuster

    def encode(self, rotor_offsets, encoded_char):

        # Handles the option for a 4th rotor
        if self.rotor_count == 4:
            output_char = self.rotor_adjuster.rotor_transition(0 - rotor_offsets[3], encoded_char)
        else:
            output_char = self.rotor_adjuster.rotor_transition(0 - rotor_offsets[2], encoded_char)

        if type(self.reflector_type) == list:
<<<<<<< HEAD:Module 1 - Principles of Programming for AI/enigma/enigma_workings/enigma.py
            # Used only for the decoding as a list is passed in instead of asingle character
=======
>>>>>>> master:Module 1 - Principles of Programming for AI/enigma/enigma.py
            output_char = self.reflector_type[EnigmaConfig.enigma_data['Alphabet'].index(output_char)]
        else:
            output_char = EnigmaConfig.enigma_data[self.reflector_type][
                EnigmaConfig.enigma_data['Alphabet'].index(output_char)]
        return output_char

    def set_rotor_count(self, count):
<<<<<<< HEAD:Module 1 - Principles of Programming for AI/enigma/enigma_workings/enigma.py
        self.rotor_count = count

    def set_reflector_type(self, reflector_type):
        self.reflector_type = reflector_type

=======
        self.rotor_count = count 

    def set_reflector_type(self, reflector_type):
        self.reflector_type = reflector_type 
 
>>>>>>> master:Module 1 - Principles of Programming for AI/enigma/enigma.py

class EnigmaMachine:
    # The main controlling class that stores the instantiated objects of the machine
    def __init__(self, position_settings, rotor_engine, rotor_turner, notch_adjuster, rotor_adjuster, reflector,
                 plugboard):
        self.position_settings = position_settings
        self.rotor_engine = rotor_engine
        self.rotor_turner = rotor_turner
        self.notch_adjuster = notch_adjuster
        self.rotor_adjuster = rotor_adjuster
        self.reflector = reflector
        self.plugboard = plugboard

    def encode(self, message, plugboard_switch):
        encoded_message = ''

        # Handles an empty string
        for character in message:
            if character == ' ':
                encoded_message += ' '
            else:
                # Toggle control for enigma with/without a plugboard
                if plugboard_switch == True:
                    character = self.plugboard.encode(character)

                # Find the adjusted positions, accouting for ring settings
                adjusted_position_settings = self.rotor_adjuster.ring_offsetter(self.position_settings, False)
                rotor_start_positions = []
                for i in adjusted_position_settings:
                    rotor = self.rotor_adjuster.rotor_offsetter(EnigmaConfig.enigma_data['Alphabet'].index(i))
                    rotor_start_positions.append(rotor)

                # Passes adjusted starting positions and turns them, accounting for notches
                notch_data = self.notch_adjuster.adjust(rotor_start_positions)
                rotor_positions = self.rotor_turner.turn(notch_data[0], notch_data[1])

                # Encodes the character after turning the rotors
                character = self.rotor_engine.encode_controller(character, rotor_positions[0])

                # Toggle control for enigma with/without a plugboard
                if plugboard_switch == True:
                    character = self.plugboard.encode(character)

                # Stores the new positions setting ready for the next character key press
                self.position_settings = rotor_positions[1]
                encoded_message += character

        # Returns the encoded message
        return encoded_message

    def reset_position_settings(self, position_settings):
        self.position_settings = position_settings

    def set_reflector(self, reflector):
        self.reflector = reflector

<<<<<<< HEAD:Module 1 - Principles of Programming for AI/enigma/enigma_workings/enigma.py

=======
>>>>>>> master:Module 1 - Principles of Programming for AI/enigma/enigma.py
