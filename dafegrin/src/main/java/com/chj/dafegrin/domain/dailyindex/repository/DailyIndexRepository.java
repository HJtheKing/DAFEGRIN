package com.chj.dafegrin.domain.dailyindex.repository;
import com.chj.dafegrin.entity.DailyIndex;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.Optional;

@Repository
public interface DailyIndexRepository extends JpaRepository<DailyIndex, Long> {
    Optional<DailyIndex> findByDate(LocalDate date);
}
