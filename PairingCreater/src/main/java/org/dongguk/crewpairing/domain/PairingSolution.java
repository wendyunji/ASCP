package org.dongguk.crewpairing.domain;

import lombok.*;
import org.dongguk.common.domain.AbstractPersistable;
import org.optaplanner.core.api.domain.solution.PlanningEntityCollectionProperty;
import org.optaplanner.core.api.domain.solution.PlanningScore;
import org.optaplanner.core.api.domain.solution.PlanningSolution;
import org.optaplanner.core.api.domain.solution.ProblemFactCollectionProperty;
import org.optaplanner.core.api.domain.valuerange.ValueRangeProvider;
import org.optaplanner.core.api.score.buildin.hardsoftlong.HardSoftLongScore;

import java.text.NumberFormat;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.ConcurrentNavigableMap;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@PlanningSolution
public class PairingSolution extends AbstractPersistable {
    //Aircraft 에 대한 모든 정보
    @ProblemFactCollectionProperty
    private List<Aircraft> aircraftList;

    //Airport 에 대한 모든 정보
    @ProblemFactCollectionProperty
    private List<Airport> airportList;

    //비행편에 대한 모든 정보 / 변수로서 작동 되므로 ValueRangeProvider 필요
    @ValueRangeProvider(id = "pairing")
    @ProblemFactCollectionProperty
    private List<Flight> flightList;

    // solver 가 풀어낸 Entity 들
    @Setter
    @PlanningEntityCollectionProperty
    private List<Pairing> pairingList;

    //score 변수
    @PlanningScore
    private HardSoftLongScore score;

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        builder.append("\n").append("Score = ").append(score).append("\n");
        for (Pairing pairing : pairingList) {
            String str = "";

            if (pairing.getPair().size() == 0) {
                str = " ---------------- !! Not Using";
            } else if (pairing.isDeadhead()) {
                str = " ---------------- !! DeadHead";
            }

            builder.append(pairing.toString()).append(str)
                    .append("\n\t\t").append(date2String(pairing)).append("\n");
        }

        return builder.toString();
    }

    @Builder
    public PairingSolution(long id, List<Aircraft> aircraftList, List<Airport> airportList, List<Flight> flightList, List<Pairing> pairingList) {
        super(id);
        this.aircraftList = aircraftList;
        this.airportList = airportList;
        this.flightList = flightList;
        this.pairingList = pairingList;
        this.score = null;
    }


    private String date2String(Pairing pairing) {
        StringBuilder sb = new StringBuilder();
        sb.append("[ ");
        for(Flight flight : pairing.getPair()){
            sb.append(" -> ").append(flight.getOriginTime()).append(" ~ ").append(flight.getDestTime());
        }
        sb.append(" ]");

        return sb.toString();
    }
    public int[] calculateMandays() {
        List<Pairing> pairingList = new ArrayList<>(this.pairingList);
        pairingList.removeIf(pairing -> pairing.getPair().isEmpty());

        List<Pairing> toRemove = new ArrayList<>();
        for (Pairing pairing : pairingList) {
            if (pairing.isNotDepartBase()) {
                toRemove.add(pairing);
            }
        }
        pairingList.removeAll(toRemove);
        pairingList.sort(Comparator.comparing(a -> a.getPair().get(0).getOriginTime()));

        int mandays = 0;
        int deactivated = 0;
        for (int i = 0; i < pairingList.size(); i++) {
            List<Flight> pair = pairingList.get(i).getPair();

            if (pairingList.get(i).isDeadhead()) {
                boolean dhComplete = false;
                for (int j = i + 1; j < pairingList.size(); j++) {
                    List<Flight> checkPair = pairingList.get(j).getPair();
                    Flight oriFlight = pair.get(0);
                    Flight dhFlight = pair.get(pair.size() - 1);
                    Flight returnFlight = checkPair.get(checkPair.size() - 1);

                    if (returnFlight.getOriginTime().isAfter(dhFlight.getDestTime())) continue;
                    if (ChronoUnit.DAYS.between(oriFlight.getOriginTime(), returnFlight.getDestTime()) > 4) continue;
                    if (!returnFlight.getOriginAirport().equals(dhFlight.getDestAirport())) continue;
                    if (!returnFlight.getDestAirport().equals(oriFlight.getOriginAirport())) continue;

                    mandays += ChronoUnit.DAYS.between(oriFlight.getOriginTime(), returnFlight.getDestTime()) + 1;
                    dhComplete = true;
                    break;
                }
                if (!dhComplete) deactivated += 1;
            }

            else mandays += ChronoUnit.DAYS.between(pair.get(0).getOriginTime(), pair.get(pair.size() - 1).getDestTime()) + 1;
    }
        return new int[] {deactivated, mandays};
    }

    public void printScore(){
        pairingList.removeIf(x -> x.getPair().isEmpty());
        int layoverTime = 360;
        int quickTurnTime = 30;

        int dutyCnt = 0;
        int deadheadCnt = 0;
        int totalLayoverTime = 0;
        int totalPairingDuration = 0;
        int dutyDays = 0;
        int activeBlockTime = 0;
        int hotelTransportationCost = 0;
        int quickTurnCost = 0;
        int layoverCost = 0;
        int satisCost = 0;
        int hotelCost = 0;
        Map<Aircraft, Integer> layoverMap = new HashMap<>();
        Map<Aircraft, Integer> quickTurnMap = new HashMap<>();
        Map<Aircraft, Integer> satisMap = new HashMap<>();
        Map<Airport, Integer> hotelMap = new HashMap<>();

        for(Pairing pairing : pairingList){
            List<Flight> pair = pairing.getPair();

            if(pair.get(0).getOriginAirport() != pair.get(pair.size()-1).getDestAirport()) deadheadCnt++; // #of deadheads
            totalPairingDuration += ChronoUnit.MINUTES.between(
                    pair.get(0).getOriginTime(), pair.get(pair.size()-1).getDestTime()); // 스케줄 전체 시간

            dutyDays += ChronoUnit.DAYS.between(pair.get(0).getOriginTime(), pair.get(pair.size()-1).getDestTime()); // 스케줄 전체 일수

            for(Flight flight : pair){
                activeBlockTime += flight.getFlightTime(); // 비행 시간만
            }

            for(int i = 0; i<pair.size()-1; i++){
                Flight flight = pair.get(i);
                Flight nextFlight = pair.get(i+1);
                int flightTime = (int) ChronoUnit.MINUTES.between(flight.getDestTime(), nextFlight.getOriginTime());
                System.out.println(flight.getDestTime()+" "+ nextFlight.getOriginTime()+" "+ flightTime);

                /*###############
                    Layover & Hotel Cost 계산
                ###############*/
                if(flightTime >= layoverTime){
                    dutyCnt++; // # of Duties
                    totalLayoverTime += flightTime; // Total Overnight Duration(레이오버 시간)
                    layoverCost += flight.getAircraft().getLayoverCost()*(flightTime-layoverTime);
                    if(layoverMap.containsKey(flight.getAircraft())){
                        layoverMap.put(flight.getAircraft(), (flightTime-layoverTime)+layoverMap.get(flight.getAircraft()));
                    }
                    else layoverMap.put(flight.getAircraft(), flightTime-layoverTime);

                    int hotelCnt = (int) (1 + Math.max(0, ChronoUnit.DAYS.between(
                            LocalDate.from(flight.getDestTime()), LocalDate.from(nextFlight.getOriginTime())) - 1));
                    hotelCost += flight.getDestAirport().getHotelCost() * hotelCnt;
                    if(hotelMap.containsKey(flight.getDestAirport())){
                        hotelMap.put(flight.getDestAirport(), hotelCnt + hotelMap.get(flight.getDestAirport()));
                    }
                    else hotelMap.put(flight.getDestAirport(), hotelCnt);
                }
                /*###############
                    Quickturn Cost 계산
                ###############*/
                else if(flightTime <= quickTurnTime){
                    quickTurnCost += flight.getAircraft().getQuickTurnCost();
                    if(quickTurnMap.containsKey(flight.getAircraft())){
                        quickTurnMap.put(flight.getAircraft(), 1+quickTurnMap.get(flight.getAircraft()));
                    }
                    else quickTurnMap.put(flight.getAircraft(), 1);
                }

                /*###############
                    Satis Cost 계산
                ###############*/
                else {
                    int startCost = pair.get(i).getAircraft().getQuickTurnCost();
                    int plusCost = startCost/(layoverTime-quickTurnTime)*(-flightTime+layoverTime);
                    satisCost += plusCost;
                    if(satisMap.containsKey((flight.getAircraft()))){
                        satisMap.put(flight.getAircraft(), (-flightTime+layoverTime)+satisMap.get(flight.getAircraft()));
                    }
                    else satisMap.put(flight.getAircraft(), (-flightTime+layoverTime));
                }
            }

        }
        NumberFormat numberFormat = NumberFormat.getNumberInstance(Locale.US);

        System.out.println("# of Pairs: " + numberFormat.format(pairingList.size()));
        System.out.println("# of Duties: " + numberFormat.format(dutyCnt + flightList.size()));
        System.out.println("# of Flight Legs: " + numberFormat.format(flightList.size()));
        System.out.println("# of Deadheads: " + numberFormat.format(deadheadCnt));
        System.out.println("# of Duty Days: " + numberFormat.format(dutyDays));

        System.out.println("Total Active Block Time: " + numberFormat.format(activeBlockTime / 60) + " h");
        System.out.println("Total Duty time: " + numberFormat.format((totalPairingDuration - totalLayoverTime) / 60) + " h");
        System.out.println("Total Overnight Duration: " + numberFormat.format(totalLayoverTime / 60) + " h");
        System.out.println("Total Pairing Duration: " + numberFormat.format(totalPairingDuration / 60) + " h");
        System.out.println("Active Block Time Per Duty Day: " + numberFormat.format((activeBlockTime / dutyDays) / 60));

        /*###############
            Hotel Transportation Cost 계산
         ###############*/
        System.out.println("\nHotel Transportation Cost: " + numberFormat.format(hotelTransportationCost));

        /*###############
            Layover Cost 계산
         ###############*/
        System.out.println("\nLayover Cost: " + numberFormat.format(layoverCost));
        for(Aircraft aircraft : layoverMap.keySet()){
            System.out.printf(aircraft.getType() +
                    "(["+aircraft.getLayoverCost()+"]*" + layoverMap.get(aircraft) +
                    "=" + aircraft.getLayoverCost()*layoverMap.get(aircraft)+"), ");
        }
        int sum = 0;
        System.out.println();
        for(Aircraft aircraft : layoverMap.keySet()){
            System.out.printf("+"+aircraft.getLayoverCost()*layoverMap.get(aircraft));
            sum += aircraft.getLayoverCost()*layoverMap.get(aircraft);
        }
        System.out.println(" = ["+ sum+"]\n");

        /*###############
            QuickTurn Cost 계산
         ###############*/
        System.out.println("QuickTurn Cost: " + numberFormat.format(quickTurnCost));
        for(Aircraft aircraft : quickTurnMap.keySet()){
            System.out.printf(aircraft.getType() + "(["+aircraft.getQuickTurnCost()+
                    "]*" + quickTurnMap.get(aircraft) +
                    "="+ aircraft.getQuickTurnCost()*quickTurnMap.get(aircraft)+"), ");
        }

        sum = 0;
        System.out.println();
        for(Aircraft aircraft : quickTurnMap.keySet()){
            System.out.printf("+"+aircraft.getQuickTurnCost()*quickTurnMap.get(aircraft));
            sum += aircraft.getQuickTurnCost()*quickTurnMap.get(aircraft);
        }
        System.out.println(" = ["+ sum+"]\n");

        /*###############
           Satis Cost 계산
         ###############*/
        System.out.println("Satis Cost: " + numberFormat.format(satisCost));
        for(Aircraft aircraft : satisMap.keySet()){
            System.out.printf(aircraft.getType() + "(["+aircraft.getQuickTurnCost()+
                    "]/[" +(layoverTime-quickTurnTime)+
                    "]*" +satisMap.get(aircraft) +
                    "="+aircraft.getQuickTurnCost()/(layoverTime-quickTurnTime)*satisMap.get(aircraft) +"), ");
        }

        sum = 0;
        System.out.println();
        for(Aircraft aircraft : satisMap.keySet()){
            System.out.printf("+"+aircraft.getQuickTurnCost()/(layoverTime-quickTurnTime)*satisMap.get(aircraft));
            sum += aircraft.getQuickTurnCost()/(layoverTime-quickTurnTime)*satisMap.get(aircraft);
        }
        System.out.println(" = ["+ sum +"]\n");

        /*###############
           Hotel Cost 계산
         ###############*/
        System.out.println("Hotel Cost: " + numberFormat.format(hotelCost));
        for(Airport airport : hotelMap.keySet()){
            System.out.printf(airport.getName() + "(["+airport.getHotelCost()+
                    "]*" + hotelMap.get(airport) +
                    "="+ airport.getHotelCost()*hotelMap.get(airport)+"), ");
        }

        sum = 0;
        System.out.println();
        for(Airport airport : hotelMap.keySet()){
            System.out.printf("+"+airport.getHotelCost()*hotelMap.get(airport));
            sum += airport.getHotelCost()*hotelMap.get(airport);
        }
        System.out.println(" = ["+ sum+"]\n");
    }
}
