import crowdai
import mido

midi_file_path="C:\\Users\\Deeps\\Documents\\School\\MIT807\\Code\\output\\sixtEncoDeco.mid"
API_KEY="<2a428240c13c65e18cea6b8a2f73ef03>"

midifile = mido.MidiFile(midi_file_path)
assert midifile.length >20 - 10 and midifile.length < 3600 + 10
assert len(midifile.tracks) == 1
assert midifile.type == 0

challenge = crowdai.Challenge("AIGeneratedMusicChallenge", API_KEY)
challenge.submit(midi_file_path)
"""
  Common pitfall: `challenge.submit` takes the `midi_file_path`
                    and not the `midifile` object
"""


