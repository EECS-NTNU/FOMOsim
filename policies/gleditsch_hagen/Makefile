compile:
	javac -cp jars/gs-core-1.3/gs-core-1.3.jar:jars/gs-algo-1.3/gs-algo-1.3.jar:jars/gs-ui-1.3/gs-ui-1.3.jar:jars/javafx-sdk-18/lib/javafx.graphics.jar:jars/poi-bin-5.2.2/poi-5.2.2.jar:jars/poi-bin-5.2.2/poi-ooxml-5.2.2.jar:jars/json-20220320.jar:/opt/xpressmp/lib/xprm.jar org/gleditsch_hagen/classes/*.java org/gleditsch_hagen/enums/*.java org/gleditsch_hagen/functions/*.java org/gleditsch_hagen/main/*.java org/gleditsch_hagen/solutionMethods/*.java org/gleditsch_hagen/xpress/*.java

jar:
	jar cvfe jars/gleditsch_hagen.jar main.Run org/gleditsch_hagen/classes/*.class org/gleditsch_hagen/enums/*.class org/gleditsch_hagen/functions/*.class org/gleditsch_hagen/main/*.class org/gleditsch_hagen/solutionMethods/*.class org/gleditsch_hagen/xpress/*.class

run:
	java -cp jars/gs-core-1.3/gs-core-1.3.jar:jars/gs-algo-1.3/gs-algo-1.3.jar:jars/gs-ui-1.3/gs-ui-1.3.jar:jars/javafx-sdk-18/lib/javafx.graphics.jar:jars/poi-bin-5.2.2/poi-5.2.2.jar:jars/poi-bin-5.2.2/poi-ooxml-5.2.2.jar:jars/json-20220320.jar:/opt/xpressmp/lib/xprm.jar:jars/gleditsch_hagen.jar org.gleditsch_hagen.main.Run
