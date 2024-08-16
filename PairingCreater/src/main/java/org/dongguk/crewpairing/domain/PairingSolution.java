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
        int pairingSize = pairingList.size();
        int deactDeadheads = 0;
        for (int i = 0; i < pairingList.size(); i++) {
            List<Flight> pair = pairingList.get(i).getPair();

            if (pairingList.get(i).isDeadhead()) {
                boolean dhComplete = false;
                for (int j = i + 1; j < pairingList.size(); j++) {
                    List<Flight> checkPair = pairingList.get(j).getPair();
                    Flight oriFlight = pair.get(0);
                    Flight dhFlight = pair.get(pair.size() - 1);
                    Flight returnFlight = checkPair.get(checkPair.size() - 1);

                    if (dhFlight.getDestTime().isAfter(returnFlight.getOriginTime())) continue;
                    if (ChronoUnit.DAYS.between(oriFlight.getOriginTime(), returnFlight.getDestTime()) > 4) continue;
                    if (!returnFlight.getOriginAirport().equals(dhFlight.getDestAirport())) continue;
                    if (!returnFlight.getDestAirport().equals(oriFlight.getOriginAirport())) continue;

                    Pairing dhPair = new Pairing();
                    dhPair.setPair(new ArrayList<>(pair));
                    dhPair.getPair().add(returnFlight);
                    if(dhPair.isImpossibleContinuity()) continue;

                    mandays += ChronoUnit.DAYS.between(oriFlight.getOriginTime().toLocalDate(), returnFlight.getDestTime().toLocalDate()) + 1;
                    dhComplete = true;
                    break;
                }
                if (!dhComplete) {
                    deactivated += pair.size();
                    deactDeadheads += 1;
                }
            }

            else mandays += ChronoUnit.DAYS.between(pair.get(0).getOriginTime().toLocalDate(), pair.get(pair.size() - 1).getDestTime().toLocalDate()) + 1;
    }
        return new int[] {pairingSize, deactivated, mandays, deactDeadheads};
    }
}
