gharrafa_taz.xml

*********************************
From SumoDOC:
 https://sumo.dlr.de/docs/Demand/Importing_O/D_Matrices.html
"A traffic assignment zone (or traffic analysis zone), short TAZ is described by its id (an arbitrary name) and lists of source and destination edges."
To distinguish the set of source and sink edges (or their probabilities respectively) use the following definition:

<tazs>
    <taz id="<TAZ_ID>">
      <tazSource id="<EDGE_ID>" weight="<PROBABILITY_TO_USE>"/>
      ... further source edges ...

      <tazSink id="<EDGE_ID>" weight="<PROBABILITY_TO_USE>"/>
      ... further destination edges ...
    </taz>

    ... further traffic assignment zones (districts) ...

</tazs>

A TAZ should have at least one source and one destination edge, each described by its id and use probability called weight herein. These edges are used to insert and remove vehicles into/from the network respectively. The probability sums of each the source and the destination lists are normalized after loading.
*********************************
TAZ file for the Gharrafa roundabout:

TAZ stands for Traffic Analysis Zone.
The Total number of TAZ in the Gharrafa model is 14, from 00 to 13
In sumo the traffic originates and end in a link.
tazSource is a source link inside the TAZ
tazSink is a destination link inside the TAZ

each Source and Sink link has a weight as a probability to be a source(destination) of traffic for the TAZ.
the total weight of all Source (Sinks) in one TAZ is 1 

<taz id="00" shape="2354.85,2254.2 2354.85,2307.56 2391.97,2307.56 2391.97,2254.2 2354.85,2254.2">
	<tazSource id="7617" weight ="0.8"/>
	<tazSource id="7618" weight ="0.2"/>
	<tazSink id="7359" weight = "1"/>
</taz>

Total Source weight=0.8+0.2
Total Sink weight=1

For taz with id=1,
the link with id="7617" has weight 0.8 (originates 80% of the traffic originating from the taz)
the link with id="7618" has weight 0.2 (originates 20% of the traffic originating from the taz)
the link with id="7359" has weight 1(originates 100% of the traffic with destination into the taz)
