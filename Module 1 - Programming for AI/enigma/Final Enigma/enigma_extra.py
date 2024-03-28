from enigma import *
from mpmath import mp


class EnigmaMachineBackPlugboardToggle:
    def __init__(self, position_settings, rotor_engine, rotor_turner, notch_adjuster, rotor_adjuster, reflector, plugboard, plugboard_toggle):
        self.position_settings = position_settings
        self.rotor_engine = rotor_engine
        self.rotor_turner = rotor_turner
        self.notch_adjuster = notch_adjuster
        self.rotor_adjuster = rotor_adjuster
        self.reflector = reflector
        self.plugboard = plugboard
        self.plugboard_toggle = plugboard_toggle

    def encode(self, message, plugboard_switch):
        encoded_message = ''
        for character in message:
            if character == ' ':
                encoded_message += ' '
            else:
                if plugboard_switch == True:
                    character = self.plugboard.encode(character)
                adjusted_position_settings = self.rotor_adjuster.ring_offsetter(self.position_settings, False)
                rotor_start_positions = []
                for i in adjusted_position_settings:
                    rotor = self.rotor_adjuster.rotor_offsetter(EnigmaConfig.enigma_data['Alphabet'].index(i))
                    rotor_start_positions.append(rotor)
                notch_data = self.notch_adjuster.adjust(rotor_start_positions)
                rotor_positions = self.rotor_turner.turn(notch_data[0], notch_data[1])
                character = self.rotor_engine.encode_controller(character, rotor_positions[0])
                if plugboard_switch == True and self.plugboard_toggle == False:

                    character = self.plugboard.encode(character)
                
                # Toggle the back plugboard on/off per keypress to enable a character to become itself 
                if self.plugboard_toggle == True:
                    self.plugboard_toggle = False
                else:
                    self.plugboard_toggle = True

                self.position_settings = rotor_positions[1]
                encoded_message += character
        return encoded_message



 
class PlugboardRotor:
    def __init__(self):
        self.leads = []
        self.rotor_idx = 0

    def add(self, plug_lead):
        self.leads.append(plug_lead)

    def encode(self, character):
        ''' Loops through PlugLead objects appended to the Plugboard and returns the encoding 
            If letter present in a PlugLead return encoded character, unless the PlugLead == the running count.
            A count that goes from 0 to len(PlugLead) and wraps back to 0
            If not, returns the same character  '''
    
        for lead in self.leads:
            encoded_character = lead.encode(character)
            if character != encoded_character:
                if self.leads.index(lead) == self.rotor_idx:
                    self.increment_rotor_count()
                    return character 
                self.increment_rotor_count()
                return encoded_character
        self.increment_rotor_count()
        return character

    def increment_rotor_count(self):
        if self.rotor_idx ==  len(self.leads):
            self.rotor_idx = 0
        else:
            self.rotor_idx += 1




class EnigmaMachinePIOffsetter:
    # The main controlling class that stores the instantiated objects of the machine 
    def __init__(self, position_settings, rotor_engine, rotor_turner, notch_adjuster, rotor_adjuster, reflector, plugboard):
        self.position_settings = position_settings
        self.rotor_engine = rotor_engine
        self.rotor_turner = rotor_turner
        self.notch_adjuster = notch_adjuster
        self.rotor_adjuster = rotor_adjuster
        self.reflector = reflector
        self.plugboard = plugboard
        self.char_count = 0

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

                #Encodes the character after turning the rotors 
                character = self.rotor_engine.encode_controller(character, rotor_positions[0])

                # Toggle control for enigma with/without a plugboard
                if plugboard_switch == True:
                    character = self.plugboard.encode(character)

                # Stores the new positions setting ready for the next character key press
                self.position_settings = rotor_positions[1]


                ''' As the encoded character exits the plugboard it enters a character shuffler based upon Pi. The nth number of Pi is calculated
                N represents the index of the character in the encoded string. This offset is then applied to the character index.
                ie the 5th character will have a char_count = 5, 5 will pass into compute_pi, and the 5th decimal place of Pi will be returned,
                This will be the offset of the outputted character. ie A becomes F. 
                '''
                pi_offset = self.compute_pi(self.char_count)
                character = self.rotor_adjuster.rotor_transition(pi_offset, character)
                encoded_message += character

        # Returns the encoded message
        return encoded_message

    def reset_position_settings(self, position_settings):
        self.position_settings = position_settings

    def set_reflector(self, reflector):
        self.reflector = reflector

    def compute_pi(self, decimal_places):
        self.char_count += 1
        if decimal_places == 0:
            return 3 
        mp.dps = decimal_places + 2  # Ensure you have enough decimal places
        pi_value = mp.pi 
        return int(str(pi_value).split('.')[-1][-2])  # Change [-1] to [-2] to get the second last digit


