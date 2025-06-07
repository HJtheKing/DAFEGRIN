package com.chj.dafegrin.domain.news.repository;

import com.chj.dafegrin.entity.News;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface NewsRepository extends JpaRepository<News, Long> {
    List<News> findAllByDailyIndex_Id(Long dailyIndexId);
}
