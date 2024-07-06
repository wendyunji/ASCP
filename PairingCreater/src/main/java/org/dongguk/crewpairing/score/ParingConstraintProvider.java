package org.dongguk.crewpairing.score;

import com.github.javaparser.utils.Pair;
import org.dongguk.crewpairing.domain.Aircraft;
import org.dongguk.crewpairing.domain.Flight;
import org.dongguk.crewpairing.domain.Pairing;
import org.optaplanner.core.api.score.buildin.hardsoftlong.HardSoftLongScore;
import org.optaplanner.core.api.score.stream.Constraint;
import org.optaplanner.core.api.score.stream.ConstraintFactory;
import org.optaplanner.core.api.score.stream.ConstraintProvider;

import java.util.List;

public class ParingConstraintProvider implements ConstraintProvider {

    //시간 최소시간
    @Override
    public Constraint[] defineConstraints(ConstraintFactory constraintFactory) {
        return new Constraint[]{
                timePossible(constraintFactory),
                airportPossible(constraintFactory),
                // aircraftPossible(constraintFactory),
                continuityPossible(constraintFactory),
                pairLengthPossible(constraintFactory),
                departBase(constraintFactory),
                deadHeadCost(constraintFactory),
                activeTimeCost(constraintFactory),
                notDepartBaseMoreThanTwo(constraintFactory)

                //layoverCost(constraintFactory),
                //quickTurnCost(constraintFactory),
                //hotelCost(constraintFactory),
                //satisCost(constraintFactory),
        };
    }

    /**
     * HARD
     * 시간 제약(Flight possible):
     * TimeImpossible 어긴 제약 -> 하드스코어 부여(1)
     */
    private Constraint timePossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(Pairing::isImpossibleTime)
                .penalize(HardSoftLongScore.ofHard(1))
                .asConstraint("Flight possible");
    }

    /**
     * HARD
     * 공간 제약(Airport possible):
     * AirportImpossible 어긴 제약 -> 하드스코어 부여(1)
     */
    private Constraint airportPossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(Pairing::isImpossibleAirport)
                .penalize(HardSoftLongScore.ofHard(1))
                .asConstraint("Airport possible");
    }

    /**
     * HARD
     * 연속된 비행 일수 제약(law possible):
     * 연속된 비행이ㅣ 14시간 이상인 제약 -> 하드스코어 부여(1)
     */
    private Constraint continuityPossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().size() >= 2)
                .filter(Pairing::isImpossibleContinuity)
                .penalize(HardSoftLongScore.ofHard(1))
                .asConstraint("law possible");
    }

    /**
     * HARD
     * 페어링 최대 4일 제약 (pairing length):
     * pairings usually take 1–4 days -> 하드스코어 부여(1)
     */
    private Constraint pairLengthPossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(Pairing::isLenghtPossible)
                .penalize(HardSoftLongScore.ofHard(1))
                .asConstraint("length possible");
    }

    /**
     * HARD
     * 모기지에서 출발하지 않는 페어링의 비행 횟수가 2를 넘어가는지 확인
     * @return Constraint
     */
    private Constraint notDepartBaseMoreThanTwo(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(Pairing::isNotDepartBaseMoreThanTwo)
                .penalize(HardSoftLongScore.ofHard(1))
                .asConstraint("Not depart base more than two flights");
    }

    /**
     * SOFT
     * 모기지에서 출발하는지 확인(Depart base):
     * Depart Base 어긴 제약 -> 소프트스코어 부여(출발지가 모기지가 아닌 페어링 당 90점)
     */
    private Constraint departBase(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(Pairing::isNotDepartBase)
                .penalize(HardSoftLongScore.ofSoft(90))
                .asConstraint("Depart base");
    }

    /**
     * SOFT
     * 페어링의 총 일수 확인(Total Mandays):
     * 총 Manday가 큰 경우 -> 소프트스코어 부여(총 Manday 일당 2점)
     */
    private Constraint activeTimeCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .penalize(HardSoftLongScore.ONE_SOFT, pairing -> 2 * pairing.getActiveTimeCost())
                .asConstraint("length cost");
    }

    /**
     * SOFT
     * deadhead cost 계산(Base diff):
     * 첫 출발공항과 마지막 도착공항이 다를 시 - > 소프트스코어 부여(데드헤드가 존재하는 페어링 당 8점)
     * @return getDeadheadCost
     */
    private Constraint deadHeadCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 1 && pairing.isEqualBase()))
                .penalize(HardSoftLongScore.ofSoft(8))
                .asConstraint("Base diff");
    }

    /**
     * ???
     * 총 layover cost 계산(Layover cost):
     * 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(layover 발생 시 cost+)
     * @return getLayoverCost
     */
    private Constraint layoverCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .penalize(HardSoftLongScore.ofSoft(0))
                // .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getLayoverCost)
                .asConstraint("Layover cost");
    }

    /**
     * ???
     * 총 QuickTurn cost 계산(QuickTurn cost):
     * 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(QuickTurn cost 발생 시 cost+)
     * @return getMovingWorkCost
     */
    private Constraint quickTurnCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .penalize(HardSoftLongScore.ofSoft(0))
                // .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getQuickTurnCost)
                .asConstraint("QuickTurn Cost");
    }

    /**
     * ???
     * 총 호텔숙박비 cost 계산(Hotel cost):
     * 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(Hotel cost 발생 시 cost+)
     * @return getHotelCost
     */
    private Constraint hotelCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .penalize(HardSoftLongScore.ofSoft(0))
                // .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getHotelCost)
                .asConstraint("Hotel Cost");
    }

    /**
     * ???
     * 승무원 만족도 cost 계산(Satis cost):
     * 승무원의 휴식시간에 따른 만족도를 코스트로 score 부여
     * / 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(Satis cost 발생 시 cost+)
     * @return getMovingWorkCost
     */
    private Constraint satisCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .penalize(HardSoftLongScore.ofSoft(0))
                // .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getSatisCost)
                .asConstraint("Satis cost");
    }
}
