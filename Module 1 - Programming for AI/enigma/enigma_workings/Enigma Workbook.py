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
        for lead in self.leads:
            encoded_character = lead.encode(character)
            if character != encoded_character:
                return encoded_character
        return character


def offset_rotor_positions(position_offset):
    rotor_order = []
    wrap_characters = False
    wrap_index = 0
    for i in range(26):
        if i + position_offset == 26:
            wrap_characters = True
        if wrap_characters == True:
            character = EnigmaConfig.enigma_data['Alphabet'][wrap_index]
            rotor_order.append(character)
            wrap_index += 1
        else:
            character = EnigmaConfig.enigma_data['Alphabet'][i + position_offset]
            rotor_order.append(character)
    return rotor_order


def set_ring_settings(ring_settings, position_settings, reversed):
    if reversed == True:
        ring_settings_reverse = []
        for i in ring_settings:
            ring_settings_reverse.append(26 - i + 2)
        ring_settings = ring_settings_reverse
    settings_list = []
    offset_position_settings = []
    for i in range(len(position_settings)):
        setting = (ring_settings[i] - 1, position_settings[i])
        settings_list.append(setting)
    for i in settings_list:
        if i[1] != 'None':
            character_idx = EnigmaConfig.enigma_data['Alphabet'].index(i[1])
            ring_offset = i[0]
            offset_idx = character_idx - ring_offset
            if offset_idx < 0:
                offset_idx = 26 + offset_idx
                offset_character = EnigmaConfig.enigma_data['Alphabet'][offset_idx]
                offset_position_settings.append(offset_character)
            else:
                offset_character = EnigmaConfig.enigma_data['Alphabet'][offset_idx]
                offset_position_settings.append(offset_character)
        else:
            offset_position_settings.append(i[1])
    return offset_position_settings


def rotor_transition(rotor_transition_offset, char):
    char_idx = EnigmaConfig.enigma_data['Alphabet'].index(char) + rotor_transition_offset
    char = EnigmaConfig.enigma_data['Alphabet'][char_idx % 26]
    return char


class Notch_Adjuster():
    def __init__(self, ring_settings, rotor_types):
        self.ring_settings = ring_settings
        self.rotor_types = rotor_types

    def adjust(self, rotor_start_positions):
        rotor_positions = []
        position_notches = []
        unadjusted_notches = []
        final_position_notches = []
        for i in rotor_start_positions:
            rotor_positions.append(i[0])
        for rotor_type in rotor_types:
            for notch in EnigmaConfig.enigma_data['Notches']:
                if notch[0] == rotor_type:
                    position_notches.append(notch[0])
                    unadjusted_notches.append(notch[1])
        adjusted_notches = set_ring_settings(ring_settings, unadjusted_notches, False)
        idx = 0
        for i in position_notches:
            final_position_notches.append((i, adjusted_notches[idx]))
            idx += 1
        return final_position_notches, rotor_positions


class Rotor_Turner:
    def __init__(self, ring_settings):
        self.ring_settings = ring_settings

    def turn(self, notch_data, rotor_positions):
        switch = False
        rotor_positions.reverse()
        notch_data.reverse()

        rotor_1_idx = EnigmaConfig.enigma_data['Alphabet'].index(rotor_positions[0])
        rotor_2_idx = EnigmaConfig.enigma_data['Alphabet'].index(rotor_positions[1])
        rotor_3_idx = EnigmaConfig.enigma_data['Alphabet'].index(rotor_positions[2])

        rotor_1_char = EnigmaConfig.enigma_data['Alphabet'][rotor_1_idx % 26]
        rotor_2_char = rotor_positions[1]
        rotor_3_char = rotor_positions[2]

        if rotor_2_char == notch_data[1][1] and notch_data[1][0] not in ('Beta', 'Gamma') and not switch:
            rotor_2_idx += 1
            rotor_2_char = EnigmaConfig.enigma_data['Alphabet'][rotor_2_idx % 26]
            rotor_3_idx += 1
            rotor_3_char = EnigmaConfig.enigma_data['Alphabet'][rotor_3_idx % 26]

        if rotor_1_char == notch_data[0][1] and notch_data[0][0] not in ('Beta', 'Gamma'):
            rotor_2_idx += 1
            rotor_2_char = EnigmaConfig.enigma_data['Alphabet'][rotor_2_idx % 26]
            switch = True

        rotor_1_idx += 1
        rotor_1_char = EnigmaConfig.enigma_data['Alphabet'][rotor_1_idx % 26]

        if len(rotor_positions) == 4:
            turned_positions_char = [rotor_1_char, rotor_2_char, rotor_3_char, rotor_positions[3]]
            turned_position_idx = [rotor_1_idx, rotor_2_idx, rotor_3_idx, EnigmaConfig.enigma_data['Alphabet'].index(rotor_positions[3])]
        else:
            turned_positions_char = [rotor_1_char, rotor_2_char, rotor_3_char]
            turned_position_idx = [rotor_1_idx, rotor_2_idx, rotor_3_idx]

        turned_positions_char.reverse()
        turned_position_idx.reverse()

        turned_position_rotors = [offset_rotor_positions(i) for i in turned_position_idx]
        turned_base_positions = set_ring_settings(self.ring_settings, turned_positions_char, True)

        return turned_position_rotors, turned_base_positions

    def offsetter(rotor_transition_offset, char):
        char_idx = EnigmaConfig.enigma_data['Alphabet'].index(char) + rotor_transition_offset
        char = EnigmaConfig.enigma_data['Alphabet'][char_idx % 26]
        return char


class Rotor_Engine():
    def __init__(self, rotor, reflector):
        self.rotor = rotor
        self.reflector = reflector

    def encode(self, character_in, rotor_positions):
        rotor_positions.reverse()
        rotor_offsets = []
        for i in rotor_positions:
            rotor_offsets.append(EnigmaConfig.enigma_data['Alphabet'].index(i[0]))
        encoded_char = self.rotor.encode(rotor_positions, rotor_offsets, 0, None, character_in)
        encoded_char = self.rotor.encode(rotor_positions, rotor_offsets, 1, 'Forward', encoded_char)
        encoded_char = self.rotor.encode(rotor_positions, rotor_offsets, 2, 'Forward', encoded_char)
        if len(rotor_types) == 4:
            encoded_char = self.rotor.encode(rotor_positions, rotor_offsets, 3, 'Forward', encoded_char)
        encoded_char = self.reflector.encode(rotor_offsets, encoded_char)
        if len(rotor_types) == 4:
            encoded_char = self.rotor.encode(rotor_positions, rotor_offsets, 3, 'Reflector', encoded_char)
        if len(rotor_types) == 4:
            encoded_char = self.rotor.encode(rotor_positions, rotor_offsets, 2, 'Backward', encoded_char)
        else:
            encoded_char = self.rotor.encode(rotor_positions, rotor_offsets, 2, 'Reflector', encoded_char)
        encoded_char = self.rotor.encode(rotor_positions, rotor_offsets, 1, 'Backward', encoded_char)
        encoded_char = self.rotor.encode(rotor_positions, rotor_offsets, 0, 'Backward', encoded_char)
        output_char = EnigmaConfig.enigma_data['Alphabet'][
            (EnigmaConfig.enigma_data['Alphabet'].index(encoded_char) - EnigmaConfig.enigma_data['Alphabet'].index(rotor_positions[0][0])) % 26]
        return output_char


class Rotor():
    def __init__(self, rotor_types):
        self.rotor_types = rotor_types

    def encode(self, rotor_positions, rotor_offsets, idx, rotor_transition_offset, input_char):
        rotor_types = self.rotor_types.copy()
        rotor_types.reverse()
        rotor_position = rotor_positions[idx]
        rotor_type = rotor_types[idx]
        if rotor_transition_offset == None:
            rotor_char_idx = EnigmaConfig.enigma_data['Alphabet'].index(input_char)
            rotor_char = rotor_position[rotor_char_idx]
        elif rotor_transition_offset == 'Reflector':
            rotor_char = rotor_transition(rotor_offsets[idx], input_char)
        elif rotor_transition_offset == 'Forward':
            rotor_char = rotor_transition(rotor_offsets[idx] - rotor_offsets[idx - 1], input_char)
        else:
            if idx == 3:
                rotor_char = rotor_transition(rotor_offsets[idx], input_char)
            else:
                rotor_char = rotor_transition(rotor_offsets[idx] - rotor_offsets[idx + 1], input_char)
        for rotor in enigma_data:
            if rotor[0] == rotor_type:
                if rotor_transition_offset == 'Forward' or rotor_transition_offset == None:
                    output_char_idx = EnigmaConfig.enigma_data['Alphabet'].index(rotor_char)
                    output_char = rotor[1][output_char_idx]
                    break
                else:
                    output_char_idx = rotor[1].index(rotor_char)
                    output_char = EnigmaConfig.enigma_data['Alphabet'][output_char_idx]
                    break
        return output_char


class Reflector():
    def __init__(self, reflector_type, rotor_types):
        self.reflector_type = reflector_type
        self.rotor_types = rotor_types

    def encode(self, rotor_offsets, encoded_char):
        if len(self.rotor_types) == 4:
            output_char = rotor_transition(0 - rotor_offsets[3], encoded_char)
        else:
            output_char = rotor_transition(0 - rotor_offsets[2], encoded_char)
        if type(self.reflector_type) == list:
            reflector = self.reflector_type
            output_char = reflector[EnigmaConfig.enigma_data['Alphabet'].index(output_char)]
        else:
            for data in enigma_data:
                if data[0] == self.reflector_type:
                    reflector = data[1]
                    output_char = reflector[EnigmaConfig.enigma_data['Alphabet'].index(output_char)]
                    break
        return output_char


class EnigmaEncoder:
    def __init__(self, position_settings, ring_settings):
        self.position_settings = position_settings
        self.ring_settings = ring_settings
        # # self.plugboard = plugboard
        # self.rotor_turner = rotor_turner
        # # self.notch_adjuster = notch_adjuster
        # self.reflector = reflector
        # self.rotor = rotor
        # self.rotor_engine = rotor_engine

    def encode(self, message):
        encoded_message = ''
        for character in message:
            if character == ' ':
                encoded_message += ' '
            else:
                character = plugboard.encode(character)
                adjusted_position_settings = set_ring_settings(self.ring_settings, self.position_settings, False)
                rotor_start_positions = []
                for i in adjusted_position_settings:
                    rotor = offset_rotor_positions(EnigmaConfig.enigma_data['Alphabet'].index(i))
                    rotor_start_positions.append(rotor)
                notch_data = notch_adjuster.adjust(rotor_start_positions)
                rotor_positions = rotor_turner.turn(notch_data[0], notch_data[1])
                character = rotor_engine.encode(character, rotor_positions[0])
                character = plugboard.encode(character)
                self.position_settings = rotor_positions[1]
                encoded_message += character
        return encoded_message


message = 'ABCDE'
plugboard = Plugboard()
plugboard.add(PlugLead('PC'))
plugboard.add(PlugLead('AD'))
plugboard.add(PlugLead('FR'))
plugboard.add(PlugLead('EH'))
plugboard.add(PlugLead('OT'))
position_settings = ['A', 'A', 'A']
ring_settings = [2, 2, 2]
rotor_types = ['II', 'I', 'III']
reflector_type = 'B'
rotor_turner = Rotor_Turner(ring_settings)
notch_adjuster = Notch_Adjuster(ring_settings, rotor_types)
reflector = Reflector(reflector_type, rotor_types)
rotor = Rotor(rotor_types)
rotor_engine = Rotor_Engine(rotor, reflector)

encoder = EnigmaEncoder(position_settings, ring_settings)
encoder.encode(message)