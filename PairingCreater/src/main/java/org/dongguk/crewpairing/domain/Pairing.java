package org.dongguk.crewpairing.domain;

import lombok.*;
import org.dongguk.common.domain.AbstractPersistable;
import org.optaplanner.core.api.domain.entity.PlanningEntity;
import org.optaplanner.core.api.domain.variable.PlanningListVariable;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Getter
@Setter
@AllArgsConstructor
@RequiredArgsConstructor
@PlanningEntity
public class Pairing extends AbstractPersistable {
    //변수로서 작동 된다. Pair 는 Flight 들의 연속이므로 ListVariable 로 작동된다.
    @PlanningListVariable(valueRangeProviderRefs = {"pairing"})
    private List<Flight> pair = new ArrayList<>();
    private Integer totalCost;
    private static int briefingTime;
    private static int debriefingTime;
    private static int restTime;
    private static int LayoverTime;
    private static int QuickTurnaroundTime;
    private static int checkContinueTime = 60 * 4;
    private static int continueMaxTime = 12 * 60;
    private static int workMaxTime = 6 * 60;

    public static void setStaticTime(int briefingTime, int debriefingTime,
                                     int restTime, int LayoverTime, int QuickTurnaroundTime) {
        Pairing.briefingTime = briefingTime;
        Pairing.debriefingTime = debriefingTime;
        Pairing.restTime = restTime;
        Pairing.LayoverTime = LayoverTime + 2*60 ;      // brief와 debrief를 고려하기 위한 layover time 2시간 증가
        Pairing.QuickTurnaroundTime = QuickTurnaroundTime;
    }

    @Builder
    public Pairing(long id, List<Flight> pair, Integer totalCost) {
        super(id);
        this.pair = pair;
        this.totalCost = totalCost;
    }

    /**
     * pairing의 실행 가능 여부 확인(불가능한 경우:true)
     * / 앞 비행이 도착하지 않았는데 이후 비행이 출발했을 경우 판단
     * @return boolean
     */
    public boolean isImpossibleTime() {
        for (int i = 0; i < pair.size() - 1; i++) {
            LocalDateTime beforeDestTime = pair.get(i).getDestTime();
            LocalDateTime afterOriginTime = pair.get(i + 1).getOriginTime();

            if (beforeDestTime.isAfter(afterOriginTime)){
                return true;
            }
        }
        return false;
    }

    /**
     *  페어링 모기지 출발 여부 확인
     *  첫번째 Flight의 출발 공항이 HB1, HB2 둘 다 아니면 True 반환
     * @return boolean
     */
    public boolean isNotDepartBase(){
        if (pair.size() == 0) return false;

        String originAirport = pair.get(0).getOriginAirport().getName();
        if(!originAirport.equals("HB1") && !originAirport.equals("HB2")) return true;

        return false;
    }

    /**
     * 모기지에서 출발하지 않는 페어링의 비행 횟수가 2를 넘어가는지 확인
     * @return boolean
     */
    public boolean isNotDepartBaseMoreThanTwo() {
        if (isNotDepartBase() && pair.size() > 2) {
            return true;
        }
        return false;
    }

    /**
     * 동일 공항 출발 여부 확인
     * / 도착 공항과 출발 공항이 다를 시 true 반환
     * @return boolean
     */
    public boolean isImpossibleAirport() {
        for (int i = 0; i < pair.size() - 1; i++) {
            String beforeAirportName = pair.get(i).getDestAirport().getName();
            String afterAirportName = pair.get(i + 1).getOriginAirport().getName();

            if (!beforeAirportName.equals(afterAirportName)) {return true;}
        }
        return false;
    }

    /**
     * 페어링의 최소 휴식시간 보장 여부 검증
     * / 쉬는 시간 포함  한 duty가 14시간 이상일 시 true 반환
     * / 또는 순수 비행 시간의 합이 8시간 이상일 시 true 반환
     * @return boolean
     */
    public boolean isImpossibleContinuity(){
        int totalTime = pair.get(0).getFlightTime();
        int workTime = pair.get(0).getFlightTime();

        for(int i=1; i<pair.size(); i++){
            int flightTime = pair.get(i).getFlightTime();
            int flightGap = getFlightGap(i - 1);

            if(flightGap < QuickTurnaroundTime) return true;

            if(flightGap < checkContinueTime) {
                totalTime += flightTime + flightGap;
                workTime += flightTime;
            }
            else {
                totalTime = flightTime;
                workTime = flightTime;
            }
            if(totalTime > continueMaxTime) return true;
            if(workTime > workMaxTime) return true;
        }
        return false;
    }

    /**
     * pairing의 동일 항공기 여부 검증(현재 사용 x)
     * / 비행들의 항공기가 동일하지 않을 시 true 반환
     * @return boolean
     */
    public boolean isDifferentAircraft() {
        for (int i = 0; i < pair.size() - 1; i++) {
            String beforeAircraft = pair.get(i).getAircraft().getType();
            String afterAircraft = pair.get(i+1).getAircraft().getType();

            if (!beforeAircraft.equals(afterAircraft)) return true;
        }
        return false;
    }

    /**
     * 처음과 끝 공항의 동일 여부 확인
     * / 처음 출발 공항과 마지막 도착 공항이 다를 시 true
     * @return boolean
     */
    public boolean isDeadhead() {
        String startAirport = pair.get(0).getOriginAirport().getName();
        String endAirport = pair.get(pair.size() - 1).getDestAirport().getName();

        return !startAirport.equals(endAirport);
    }

    /**
     * 페어링의 총 SatisCost 반환
     * / breakTime이 30보다 크고 180보다 작은 경우 발생
     * / (30,퀵턴)-(180,0)을 지나는 일차함수 형태로 작성
     * @return 퀵턴코스트의 합
     */
    public Integer getSatisCost(){
        int satisScore = 0;
        for(int i=0; i<pair.size()-1; i++){
            if(getFlightGap(i) < LayoverTime && getFlightGap(i) > QuickTurnaroundTime){
                int startCost = (int) pair.get(i).getAircraft().getQuickTurnCost();
                int plusScore = (startCost/(LayoverTime-QuickTurnaroundTime))*(-getFlightGap(i)+LayoverTime);
                satisScore += plusScore;
                //(30,퀵턴)-(180,0)을 지나는 일차함수
            }
        }
        return (int) satisScore;
    }

    /**
     * 페어링의 총 길아 반환 (일)
     * @return 마지막 비행 도착시간 - 처음 비행 시작시간
     */
    public Integer getActiveTimeCost() {
        if (pair.size() == 0) return 0;

        return Math.max(0, (int) ChronoUnit.DAYS.between(pair.get(0).getOriginTime(), pair.get(pair.size() - 1).getDestTime()));
    }

    /**
     * 페어링이 4일을 넘는 지 반환 (Boolean)
     * @return 4일을 넘으면 true, 아니면 false
     */
    public Boolean isLenghtPossible(){
        if (pair.size() <= 1) return false;

        if (ChronoUnit.DAYS.between(pair.get(0).getOriginTime(), pair.get(pair.size() - 1).getDestTime())+1 > 4)
            return true;
        else return false;
    }

    /**
     * 페어링의 deadhead cost 반환
     * / 마지막 도착 공항에서 처음 공항으로 가는데 필요한 deadhead cost 사용
     * @return deadhead cost / 2
     */
    public Integer getDeadHeadCost() {
        Map<String, Integer> deadheads = pair.get(pair.size() - 1).getDestAirport().getDeadheadCost();
        String origin = pair.get(0).getOriginAirport().getName();

        return deadheads.getOrDefault(origin, 0) *3;
    }

    /**
     * 페어링의 총 LayoverCost 반환
     * 비행편간 간격이 LayoverTime 보다 크거나 같은 경우에만 LayoverCost 발생
     * @return sum(LayoverCost) / 100
     */
    public Integer getLayoverCost(){
        // 페어링의 총 길이가 1개 이하라면 LayoverCost 없음
        if(pair.size() <= 1) return 0;

        int layoverCost = pair.get(0).getAircraft().getLayoverCost();
        int cost = 0;
        for (int i = 0; i < pair.size() - 1; i++) {
            // 만약 비행편 간격이 하나라도 음수라면 유효한 페어링이 아님
            if (getFlightGap(i) <= 0) {
                return 0;
            }
            // 음수가 아니라면 유효한 페어링이므로 LayoverCost 계산
            if (getFlightGap(i) >= LayoverTime) {
                cost += (getFlightGap(i) - LayoverTime) * layoverCost;
            }
        }

        return cost;
    }

    /**
     * 페어링의 총 QuickTurnCost 반환
     * 비행편간 간격이 QuickTurnaroundTime 보다 작은 경우에만 QuickTurnCost 발생
     * 퀵턴의 경우 같은 비행기를 사용
     * @return sum(QuickTurnCost) / 100
     */
    public Integer getQuickTurnCost() {
        // 페어링의 총 길이가 1개 이하라면 QuickTurnCost 없음
        if(pair.size() <= 1) return 0;
        int quickTurnCost = pair.get(0).getAircraft().getQuickTurnCost();
        int cost = 0;

        for (int i = 0; i < pair.size() - 1; i++) {

            //비행기가 같을 때만 수행
            if (getFlightGap(i) <= QuickTurnaroundTime) {
                cost += quickTurnCost;
            }
        }

        return cost;
    }

    /**
     * 페어링의 총 HotelCost 반환
     * / 총 인원수를 곱하는 이유 : Flight Cost, Layover Cost, QuickTurn Cost 모두 총 인원에 대한 값으로 계산된 후 입력받음
     * / 휴식시간이 12시간 이상일 경우 1일 숙박,이후 18시간 이상 남을 시 1일 추가 반복
     * @return sum(hotel cost) / 100
     */
    public Integer getHotelCost() {
        // 페어링의 총 길이가 1개 이하라면 HotelCost 없음
        if(pair.size() <= 1) return 0;

        int cost = 0;
        for (int i = 0; i < pair.size() - 1; i++) {
            // 만약 비행편 간격이 하나라도 0이라면 유효한 페어링이 아님
            int flightGap = getFlightGap(i);
            if (flightGap == 0) return 0;

            LocalDate layoverStartTime = LocalDate.from(pair.get(i).getDestTime());
            LocalDate layoverFinishTime = LocalDate.from(pair.get(i+1).getOriginTime());

            //layover가 발생했으면 일단 1회 발생, 이후 날짜가 바뀔 때 마다 1회씩 발생
            if (flightGap >= LayoverTime) {
                cost += pair.get(i).getDestAirport().getHotelCost()
                        * (1 + Math.max(0, ChronoUnit.DAYS.between(layoverStartTime, layoverFinishTime) - 1));
            }
        }

        return cost;
    }

    /**
     * 비행 사이의 쉬는 시간 계산
     * @return (int) Math.max(0,breakTime)
     */
    private int getFlightGap(int index){ //수정 필요
        long breakTime = ChronoUnit.MINUTES.between(pair.get(index).getDestTime(), pair.get(index+1).getOriginTime());

        return (int) Math.max(0, breakTime);
    }

    @Override
    public String toString() {
        return "Pairing - " + id +
                " { pair=" + pair + " }";
    }
}
