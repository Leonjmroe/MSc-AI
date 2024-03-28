from enigma import *


def test_plugboard_overlapping_encoding():
    plugboard = Plugboard()
    plugboard.add(PlugLead('AB'))
    plugboard.add(PlugLead('BC'))
    
    assert plugboard.encode('A') == 'B'
    assert plugboard.encode('B') == 'A'
    assert plugboard.encode('C') == 'C'

test_plugboard_overlapping_encoding()


def test_plugboard_delete():
    plugboard = Plugboard()
    plugboard.add(PlugLead('AB'))
    plugboard.add(PlugLead('CD'))
    plugboard.add(PlugLead('EF'))
    plugboard.delete()
    
    assert plugboard.encode('A') == 'B'
    assert plugboard.encode('C') == 'D'
    assert plugboard.encode('E') == 'E'
    assert plugboard.encode('F') == 'F'

test_plugboard_delete()
